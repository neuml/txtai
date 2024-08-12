"""
Runs benchmark evaluations with the BEIR dataset.

Install txtai and the following dependencies to run:
    pip install txtai pytrec_eval rank-bm25 bm25s elasticsearch psutil
"""

import argparse
import csv
import json
import os
import pickle
import sqlite3
import time

import psutil
import yaml

import numpy as np

from bm25s import BM25 as BM25Sparse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from pytrec_eval import RelevanceEvaluator
from rank_bm25 import BM25Okapi

from txtai.embeddings import Embeddings
from txtai.pipeline import LLM, RAG, Tokenizer
from txtai.scoring import ScoringFactory


class Index:
    """
    Base index definition. Defines methods to index and search a dataset.
    """

    def __init__(self, path, config, output, refresh):
        """
        Creates a new index.

        Args:
            path: path to dataset
            config: path to config file
            output: path to store index
            refresh: overwrites existing index if True, otherwise existing index is loaded
        """

        self.path = path
        self.config = config
        self.output = output
        self.refresh = refresh

        # Build and save index
        self.backend = self.index()

    def __call__(self, limit, filterscores=True):
        """
        Main evaluation logic. Loads an index, runs the dataset queries and returns the results.

        Args:
            limit: maximum results
            filterscores: if exact matches should be filtered out

        Returns:
            search results
        """

        uids, queries = self.load()

        # Run queries in batches
        offset, results = 0, {}
        for batch in self.batch(queries, 256):
            for i, r in enumerate(self.search(batch, limit + 1)):
                # Get result as list of (id, score) tuples
                r = list(r)
                r = [(x["id"], x["score"]) for x in r] if r and isinstance(r[0], dict) else r

                if filterscores:
                    r = [(uid, score) for uid, score in r if uid != uids[offset + i]][:limit]

                results[uids[offset + i]] = dict(r)

            # Increment offset
            offset += len(batch)

        return results

    def search(self, queries, limit):
        """
        Runs a search for a set of queries.

        Args:
            queries: list of queries to run
            limit: maximum results

        Returns:
            search results
        """

        return self.backend.batchsearch(queries, limit)

    def index(self):
        """
        Indexes a dataset.
        """

        raise NotImplementedError

    def rows(self):
        """
        Iterates over the dataset yielding a row at a time for indexing.
        """

        with open(f"{self.path}/corpus.jsonl", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                text = f'{row["title"]}. {row["text"]}' if row.get("title") else row["text"]
                if text:
                    yield (row["_id"], text, None)

    def load(self):
        """
        Loads queries for the dataset. Returns a list of expected result ids and input queries.

        Returns:
            (result ids, input queries)
        """

        with open(f"{self.path}/queries.jsonl", encoding="utf-8") as f:
            data = [json.loads(query) for query in f]
            uids, queries = [x["_id"] for x in data], [x["text"] for x in data]

        return uids, queries

    def batch(self, data, size):
        """
        Splits data into equal sized batches.

        Args:
            data: input data
            size: batch size

        Returns:
            data split into equal size batches
        """

        return [data[x : x + size] for x in range(0, len(data), size)]

    def readconfig(self, key, default):
        """
        Reads configuration from a config file. Returns default configuration
        if config file is not found or config key isn't present.

        Args:
            key: configuration key to lookup
            default: default configuration

        Returns:
            config if found, otherwise returns default config
        """

        if self.config and os.path.exists(self.config):
            # Read configuration
            with open(self.config, "r", encoding="utf-8") as f:
                # Check for config
                config = yaml.safe_load(f)
                if key in config:
                    return config[key]

        return default


class Embed(Index):
    """
    Embeddings index using txtai.
    """

    def index(self):
        if os.path.exists(self.output) and not self.refresh:
            embeddings = Embeddings()
            embeddings.load(self.output)
        else:
            # Read configuration
            config = self.readconfig("embeddings", {"batch": 8192, "encodebatch": 128, "faiss": {"quantize": True, "sample": 0.05}})

            # Build index
            embeddings = Embeddings(config)
            embeddings.index(self.rows())
            embeddings.save(self.output)

        return embeddings


class Hybrid(Index):
    """
    Hybrid embeddings + BM25 index using txtai.
    """

    def index(self):
        if os.path.exists(self.output) and not self.refresh:
            embeddings = Embeddings()
            embeddings.load(self.output)
        else:
            # Read configuration
            config = self.readconfig(
                "hybrid",
                {
                    "batch": 8192,
                    "encodebatch": 128,
                    "faiss": {"quantize": True, "sample": 0.05},
                    "scoring": {"method": "bm25", "terms": True, "normalize": True},
                },
            )

            # Build index
            embeddings = Embeddings(config)
            embeddings.index(self.rows())
            embeddings.save(self.output)

        return embeddings


class RetrievalAugmentedGeneration(Embed):
    """
    Retrieval augmented generation (RAG) using txtai.
    """

    def __init__(self, path, config, output, refresh):
        # Parent logic
        super().__init__(path, config, output, refresh)

        # Read LLM configuration
        llm = self.readconfig("llm", {})

        # Read RAG configuration
        rag = self.readconfig("rag", {})

        # Load RAG pipeline
        self.rag = RAG(self.backend, LLM(**llm), output="reference", **rag)

    def search(self, queries, limit):
        # Set context window size to limit and run
        self.rag.context = limit
        return [[(x["reference"], 1)] for x in self.rag(queries, maxlength=4096)]


class Score(Index):
    """
    BM25 index using txtai.
    """

    def index(self):
        # Read configuration
        config = self.readconfig("scoring", {"method": "bm25", "terms": True})

        # Create scoring instance
        scoring = ScoringFactory.create(config)

        output = os.path.join(self.output, "scoring")
        if os.path.exists(output) and not self.refresh:
            scoring.load(output)
        else:
            scoring.index(self.rows())
            scoring.save(output)

        return scoring


class RankBM25(Index):
    """
    BM25 index using rank-bm25.
    """

    def search(self, queries, limit):
        ids, backend = self.backend
        tokenizer, results = Tokenizer(), []
        for query in queries:
            scores = backend.get_scores(tokenizer(query))
            topn = np.argsort(scores)[::-1][:limit]
            results.append([(ids[x], scores[x]) for x in topn])

        return results

    def index(self):
        output = os.path.join(self.output, "rank")
        if os.path.exists(output) and not self.refresh:
            with open(output, "rb") as f:
                ids, model = pickle.load(f)
        else:
            # Tokenize data
            tokenizer, data = Tokenizer(), []
            for uid, text, _ in self.rows():
                data.append((uid, tokenizer(text)))

            ids = [uid for uid, _ in data]
            model = BM25Okapi([text for _, text in data])

            # Save model
            with open(output, "wb") as out:
                pickle.dump(model, out)

        return ids, model


class BM25S(Index):
    """
    BM25 as implemented by bm25s
    """

    def __init__(self, path, config, output, refresh):
        # Corpus ids
        self.ids = None

        # Parent logic
        super().__init__(path, config, output, refresh)

    def search(self, queries, limit):
        tokenizer = Tokenizer()
        results, scores = self.backend.retrieve([tokenizer(x) for x in queries], corpus=self.ids, k=limit)

        # List of queries => list of matches (id, score)
        x = []
        for a, b in zip(results, scores):
            x.append([(str(c), float(d)) for c, d in zip(a, b)])

        return x

    def index(self):
        tokenizer = Tokenizer()
        ids, texts = [], []

        for uid, text, _ in self.rows():
            ids.append(uid)
            texts.append(text)

        self.ids = ids

        if os.path.exists(self.output) and not self.refresh:
            model = BM25Sparse.load(self.output)
        else:
            model = BM25Sparse(method="lucene", k1=1.2, b=0.75)
            model.index([tokenizer(x) for x in texts], leave_progress=False)
            model.save(self.output)

        return model


class SQLiteFTS(Index):
    """
    BM25 index using SQLite's FTS extension.
    """

    def search(self, queries, limit):
        tokenizer, results = Tokenizer(), []
        for query in queries:
            query = tokenizer(query)
            query = " OR ".join([f'"{q}"' for q in query])

            self.backend.execute(
                f"SELECT id, bm25(textindex) * -1 score FROM textindex WHERE text MATCH ? ORDER BY bm25(textindex) LIMIT {limit}", [query]
            )

            results.append(list(self.backend))

        return results

    def index(self):
        if os.path.exists(self.output) and not self.refresh:
            # Load existing database
            connection = sqlite3.connect(self.output)
        else:
            # Delete existing database
            if os.path.exists(self.output):
                os.remove(self.output)

            # Create new database
            connection = sqlite3.connect(self.output)

            # Tokenize data
            tokenizer, data = Tokenizer(), []
            for uid, text, _ in self.rows():
                data.append((uid, " ".join(tokenizer(text))))

            # Create table
            connection.execute("CREATE VIRTUAL TABLE textindex using fts5(id, text)")

            # Load data and build index
            connection.executemany("INSERT INTO textindex VALUES (?, ?)", data)

            connection.commit()

        return connection.cursor()


class Elastic(Index):
    """
    BM25 index using Elasticsearch.
    """

    def search(self, queries, limit):
        # Generate bulk queries
        request = []
        for query in queries:
            req_head = {"index": "textindex", "search_type": "dfs_query_then_fetch"}
            req_body = {
                "_source": False,
                "query": {"multi_match": {"query": query, "type": "best_fields", "fields": ["text"], "tie_breaker": 0.5}},
                "size": limit,
            }
            request.extend([req_head, req_body])

        # Run ES query
        response = self.backend.msearch(body=request, request_timeout=600)

        # Read responses
        results = []
        for resp in response["responses"]:
            result = resp["hits"]["hits"]
            results.append([(r["_id"], r["_score"]) for r in result])

        return results

    def index(self):
        es = Elasticsearch("http://localhost:9200")

        # Delete existing index
        # pylint: disable=W0702
        try:
            es.indices.delete(index="textindex")
        except:
            pass

        bulk(es, ({"_index": "textindex", "_id": uid, "text": text} for uid, text, _ in self.rows()))
        es.indices.refresh(index="textindex")

        return es


def relevance(path):
    """
    Loads relevance data for evaluation.

    Args:
        path: path to dataset test file

    Returns:
        relevance data
    """

    rel = {}
    with open(f"{path}/qrels/test.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        next(reader)

        for row in reader:
            queryid, corpusid, score = row[0], row[1], int(row[2])
            if queryid not in rel:
                rel[queryid] = {corpusid: score}
            else:
                rel[queryid][corpusid] = score

    return rel


def create(method, path, config, output, refresh):
    """
    Creates a new index.

    Args:
        method: indexing method
        path: path to dataset
        config: path to config file
        output: path to store index
        refresh: overwrites existing index if True, otherwise existing index is loaded

    Returns:
        Index
    """

    if method == "hybrid":
        return Hybrid(path, config, output, refresh)
    if method == "rag":
        return RetrievalAugmentedGeneration(path, config, output, refresh)
    if method == "scoring":
        return Score(path, config, output, refresh)
    if method == "rank":
        return RankBM25(path, config, output, refresh)
    if method == "bm25s":
        return BM25S(path, config, output, refresh)
    if method == "sqlite":
        return SQLiteFTS(path, config, output, refresh)
    if method == "es":
        return Elastic(path, config, output, refresh)

    # Default
    return Embed(path, config, output, refresh)


def compute(results):
    """
    Computes metrics using the results from an evaluation run.

    Args:
        results: evaluation results

    Returns:
        metrics
    """

    metrics = {}
    for r in results:
        for metric in results[r]:
            if metric not in metrics:
                metrics[metric] = []

            metrics[metric].append(results[r][metric])

    return {metric: round(np.mean(values), 5) for metric, values in metrics.items()}


def evaluate(methods, path, args):
    """
    Runs an evaluation.

    Args:
        methods: list of indexing methods to test
        path: path to dataset
        args: command line arguments

    Returns:
        {calculated performance metrics}
    """

    print(f"------ {os.path.basename(path)} ------")

    # Performance stats
    performance = {}

    # Calculate stats for each model type
    topk = args.topk
    evaluator = RelevanceEvaluator(relevance(path), {f"ndcg_cut.{topk}", f"map_cut.{topk}", f"recall.{topk}", f"P.{topk}"})
    for method in methods:
        # Stats for this source
        stats = {}
        performance[method] = stats

        # Create index and get results
        start = time.time()
        output = args.output if args.output else f"{path}/{method}"
        index = create(method, path, args.config, output, args.refresh)

        # Add indexing metrics
        stats["index"] = round(time.time() - start, 2)
        stats["memory"] = int(psutil.Process().memory_info().rss / (1024 * 1024))
        stats["disk"] = int(sum(d.stat().st_size for d in os.scandir(output) if d.is_file()) / 1024) if os.path.isdir(output) else 0

        print("INDEX TIME =", time.time() - start)
        print(f"MEMORY USAGE = {stats['memory']} MB")
        print(f"DISK USAGE = {stats['disk']} KB")

        start = time.time()
        results = index(topk)

        # Add search metrics
        stats["search"] = round(time.time() - start, 2)
        print("SEARCH TIME =", time.time() - start)

        # Calculate stats
        metrics = compute(evaluator.evaluate(results))

        # Add accuracy metrics
        for stat in [f"ndcg_cut_{topk}", f"map_cut_{topk}", f"recall_{topk}", f"P_{topk}"]:
            stats[stat] = metrics[stat]

        # Print model stats
        print(f"------ {method} ------")
        print(f"NDCG@{topk} =", metrics[f"ndcg_cut_{topk}"])
        print(f"MAP@{topk} =", metrics[f"map_cut_{topk}"])
        print(f"Recall@{topk} =", metrics[f"recall_{topk}"])
        print(f"P@{topk} =", metrics[f"P_{topk}"])

    print()
    return performance


def benchmarks(args):
    """
    Main benchmark execution method.

    Args:
        args: command line arguments
    """

    # Directory where BEIR datasets are stored
    directory = args.directory if args.directory else "beir"

    if args.sources and args.methods:
        sources, methods = args.sources.split(","), args.methods.split(",")
        mode = "a"
    else:
        # Default sources and methods
        sources = [
            "trec-covid",
            "nfcorpus",
            "nq",
            "hotpotqa",
            "fiqa",
            "arguana",
            "webis-touche2020",
            "quora",
            "dbpedia-entity",
            "scidocs",
            "fever",
            "climate-fever",
            "scifact",
        ]
        methods = ["embed", "hybrid", "rag", "scoring", "rank", "bm25s", "sqlite", "es"]
        mode = "w"

    # Run and save benchmarks
    with open("benchmarks.json", mode, encoding="utf-8") as f:
        for source in sources:
            # Run evaluations
            results = evaluate(methods, f"{directory}/{source}", args)

            # Save as JSON lines output
            for method, stats in results.items():
                stats["source"] = source
                stats["method"] = method
                stats["name"] = args.name if args.name else method

                json.dump(stats, f)
                f.write("\n")


if __name__ == "__main__":
    # Command line parser
    parser = argparse.ArgumentParser(description="Benchmarks")
    parser.add_argument("-c", "--config", help="path to config file", metavar="CONFIG")
    parser.add_argument("-d", "--directory", help="root directory path with datasets", metavar="DIRECTORY")
    parser.add_argument("-m", "--methods", help="comma separated list of methods", metavar="METHODS")
    parser.add_argument("-n", "--name", help="name to assign to this run, defaults to method name", metavar="NAME")
    parser.add_argument("-o", "--output", help="index output directory path", metavar="OUTPUT")
    parser.add_argument(
        "-r",
        "--refresh",
        help="refreshes index if set, otherwise uses existing index if available",
        action="store_true",
    )
    parser.add_argument("-s", "--sources", help="comma separated list of data sources", metavar="SOURCES")
    parser.add_argument("-t", "--topk", help="top k results to use for the evaluation", metavar="TOPK", type=int, default=10)

    # Calculate benchmarks
    benchmarks(parser.parse_args())
