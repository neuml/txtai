"""
Runs benchmark evaluations with the BEIR dataset.

Install txtai and the following dependencies to run:
    pip install txtai pytrec_eval rank-bm25 elasticsearch psutil
"""

import csv
import json
import os
import pickle
import sqlite3
import sys
import time

import psutil
import yaml

import numpy as np

from rank_bm25 import BM25Okapi
from pytrec_eval import RelevanceEvaluator

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from txtai.embeddings import Embeddings
from txtai.pipeline import Tokenizer
from txtai.scoring import ScoringFactory


class Index:
    """
    Base index definition. Defines methods to index and search a dataset.
    """

    def __init__(self, path, refresh=True):
        """
        Creates a new index.

        Args:
            path: path to dataset
            refresh: overwrites existing index if True, otherwise existing index is loaded
        """

        self.path = path
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
                r = list(r)
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
                text = f'{row["title"]}. {row["text"]}' if row["title"] else row["text"]
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

    def config(self, key, default):
        """
        Reads configuration from a config.yml file. Returns default configuration
        if config.yml file is not found or config key isn't present.

        Args:
            key: configuration key to lookup
            default: default configuration

        Returns:
            config if found, otherwise returns default config
        """

        if os.path.exists("config.yml"):
            # Read configuration
            with open("config.yml", "r", encoding="utf-8") as f:
                # Check for config
                config = yaml.safe_load(f)
                if key in config:
                    return config[key]

        return default


class Score(Index):
    """
    BM25 index using txtai.
    """

    def index(self):
        # Read configuration
        config = self.config("scoring", {"method": "bm25", "terms": True})

        # Create scoring instance
        scoring = ScoringFactory.create(config)

        path = f"{self.path}/scoring"
        if os.path.exists(path) and not self.refresh:
            scoring.load(path)
        else:
            scoring.index(self.rows())
            scoring.save(path)

        return scoring


class Embed(Index):
    """
    Embeddings index using txtai.
    """

    def index(self):
        path = f"{self.path}/embeddings"
        if os.path.exists(path) and not self.refresh:
            embeddings = Embeddings()
            embeddings.load(path)
        else:
            # Read configuration
            config = self.config("embed", {"batch": 8192, "encodebatch": 128, "faiss": {"quantize": True, "sample": 0.05}})

            # Build index
            embeddings = Embeddings(config)
            embeddings.index(self.rows())
            embeddings.save(path)

        return embeddings


class Hybrid(Index):
    """
    Hybrid embeddings + BM25 index using txtai.
    """

    def index(self):
        path = f"{self.path}/hybrid"
        if os.path.exists(path) and not self.refresh:
            embeddings = Embeddings()
            embeddings.load(path)
        else:
            # Read configuration
            config = self.config(
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
            embeddings.save(path)

        return embeddings


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
        path = f"{self.path}/rankbm25"
        if os.path.exists(path) and not self.refresh:
            with open(path, "rb") as f:
                ids, model = pickle.load(f)
        else:
            # Tokenize data
            tokenizer, data = Tokenizer(), []
            for uid, text, _ in self.rows():
                data.append((uid, tokenizer(text)))

            ids = [uid for uid, _ in data]
            model = BM25Okapi([text for _, text in data])

        return ids, model


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
        path = f"{self.path}/fts.sqlite"
        if os.path.exists(path) and not self.refresh:
            # Load existing database
            connection = sqlite3.connect(path)
        else:
            # Delete existing database
            if os.path.exists(path):
                os.remove(path)

            # Create new database
            connection = sqlite3.connect(path)

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


def create(name, path, refresh):
    """
    Creates a new index.

    Args:
        path: dataset path
        refresh: overwrites existing index if True, otherwise existing index is loaded

    Returns:
        Index
    """

    if name == "embed":
        return Embed(path, refresh)
    if name == "es":
        return Elastic(path, refresh)
    if name == "hybrid":
        return Hybrid(path, refresh)
    if name == "sqlite":
        return SQLiteFTS(path, refresh)
    if name == "rank":
        return RankBM25(path, refresh)

    # Default
    return Score(path, refresh)


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


def evaluate(path, methods):
    """
    Runs an evaluation.

    Args:
        path: path to dataset
        methods: list of indexing methods to test

    Returns:
        {calculated performance metrics}
    """

    print(f"------ {os.path.basename(path)} ------")

    # Performance stats
    performance = {}

    # Calculate stats for each model type
    topk, refresh = 10, True
    evaluator = RelevanceEvaluator(relevance(path), {f"ndcg_cut.{topk}", f"map_cut.{topk}", f"recall.{topk}", f"P.{topk}"})
    for method in methods:
        # Stats for this source
        stats = {}
        performance[method] = stats

        # Create index and get results
        start = time.time()
        index = create(method, path, refresh)

        # Add indexing metrics
        stats["index"] = round(time.time() - start, 2)
        stats["memory"] = int(psutil.Process().memory_info().rss / (1024 * 1024))

        print("INDEX TIME =", time.time() - start)
        print(f"MEMORY USAGE = {psutil.Process().memory_info().rss / (1024 * 1024)} MB")

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


def benchmarks():
    """
    Main benchmark execution method.
    """

    # Directory where BEIR datasets are stored
    directory = sys.argv[1] if len(sys.argv) > 1 else "beir"

    if len(sys.argv) > 3:
        sources = [sys.argv[2]]
        methods = [sys.argv[3]]
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
        methods = ["bm25", "embed", "es", "hybrid", "rank", "sqlite"]
        mode = "w"

    # Run and save benchmarks
    with open("benchmarks.json", mode, encoding="utf-8") as f:
        for source in sources:
            # Run evaluations
            results = evaluate(f"{directory}/{source}", methods)

            # Save as JSON lines output
            for method, stats in results.items():
                stats["source"] = source
                stats["method"] = method

                json.dump(stats, f)
                f.write("\n")


# Calculate benchmarks
benchmarks()
