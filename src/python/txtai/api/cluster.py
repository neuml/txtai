"""
Cluster module
"""

import asyncio
import urllib.parse
import zlib

import aiohttp

from ..database.sql import Aggregate


class Cluster:
    """
    Aggregates multiple embeddings shards into a single logical embeddings instance.
    """

    # pylint: disable = W0231
    def __init__(self, config=None):
        """
        Creates a new Cluster.

        Args:
            config: cluster configuration
        """

        # Configuration
        self.config = config

        # Embeddings shard urls
        self.shards = None
        if "shards" in self.config:
            self.shards = self.config["shards"]

        # Query aggregator
        self.aggregate = Aggregate()

    def search(self, query, limit=None):
        """
        Finds documents in the embeddings cluster most similar to the input query. Returns
        a list of {id: value, score: value} sorted by highest score, where id is the
        document id in the embeddings model.

        Args:
            query: query text
            limit: maximum results

        Returns:
            list of {id: value, score: value}
        """

        # Build URL
        action = f"search?query={urllib.parse.quote_plus(query)}"
        if limit:
            action += f"&limit={limit}"

        # Run query and flatten results into single results list
        results = []
        for result in self.execute("get", action):
            results.extend(result)

        # Combine aggregate functions and sort
        results = self.aggregate(query, results)

        # Limit results
        return results[: (limit if limit else 10)]

    def batchsearch(self, queries, limit=None):
        """
        Finds documents in the embeddings cluster most similar to the input queries. Returns
        a list of {id: value, score: value} sorted by highest score per query, where id is
        the document id in the embeddings model.

        Args:
            queries: queries text
            limit: maximum results

        Returns:
            list of {id: value, score: value} per query
        """

        # POST parameters
        params = {"queries": queries}
        if limit:
            params["limit"] = limit

        # Run query
        batch = self.execute("post", "batchsearch", [params] * len(self.shards))

        # Combine results per query
        results = []
        for x, query in enumerate(queries):
            result = []
            for section in batch:
                result.extend(section[x])

            # Aggregate, sort and limit results
            results.append(self.aggregate(query, result)[: (limit if limit else 10)])

        return results

    def add(self, documents):
        """
        Adds a batch of documents for indexing.

        Args:
            documents: list of {id: value, text: value}
        """

        self.execute("post", "add", self.shard(documents))

    def index(self):
        """
        Builds an embeddings index for previously batched documents.
        """

        self.execute("get", "index")

    def upsert(self):
        """
        Runs an embeddings upsert operation for previously batched documents.
        """

        self.execute("get", "upsert")

    def delete(self, ids):
        """
        Deletes from an embeddings cluster. Returns list of ids deleted.

        Args:
            ids: list of ids to delete

        Returns:
            ids deleted
        """

        return [uid for ids in self.execute("post", "delete", self.shard(ids)) for uid in ids]

    def count(self):
        """
        Total number of elements in this embeddings cluster.

        Returns:
            number of elements in embeddings cluster
        """

        return sum(self.execute("get", "count"))

    def shard(self, documents):
        """
        Splits documents into equal sized shards.

        Args:
            documents: input documents

        Returns:
            list of evenly sized shards with the last shard having the remaining elements
        """

        shards = [[] for _ in range(len(self.shards))]
        for document in documents:
            uid = document["id"] if isinstance(document, dict) else document
            if isinstance(uid, str):
                # Quick int hash of string to help derive shard id
                uid = zlib.adler32(uid.encode("utf-8"))

            shards[uid % len(self.shards)].append(document)

        return shards

    def execute(self, method, action, data=None):
        """
        Executes a HTTP action asynchronously.

        Args:
            method: get or post
            action: url action to perform
            data: post parameters

        Returns:
            json results if any
        """

        # Get urls
        urls = [f"{shard}/{action}" for shard in self.shards]
        close = False

        # Use existing loop if available, otherwise create one
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            close = True

        try:
            return loop.run_until_complete(self.run(urls, method, data))
        finally:
            # Close loop if it was created in this method
            if close:
                loop.close()

    async def run(self, urls, method, data):
        """
        Runs an async action.

        Args:
            urls: run against this list of urls
            method: get or post
            data: list of data for each url or None

        Returns:
            json results if any
        """

        async with aiohttp.ClientSession(raise_for_status=True) as session:
            tasks = []

            for x, url in enumerate(urls):
                if method == "post":
                    if not data or data[x]:
                        tasks.append(asyncio.ensure_future(self.post(session, url, data[x] if data else None)))
                else:
                    tasks.append(asyncio.ensure_future(self.get(session, url)))

            return await asyncio.gather(*tasks)

    async def get(self, session, url):
        """
        Runs an async HTTP GET request.

        Args:
            session: ClientSession
            url: url

        Returns:
            json results if any
        """

        async with session.get(url) as resp:
            return await resp.json()

    async def post(self, session, url, data):
        """
        Runs an async HTTP POST request.

        Args:
            session: ClientSession
            url: url
            data: data to POST

        Returns:
            json results if any
        """

        async with session.post(url, json=data) as resp:
            return await resp.json()
