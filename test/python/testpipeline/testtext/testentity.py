"""
Entity module tests
"""

import unittest

from txtai.pipeline import Entity


class TestEntity(unittest.TestCase):
    """
    Entity tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create entity instance.
        """

        cls.entity = Entity("dslim/bert-base-NER")

    def testEntity(self):
        """
        Test entity
        """

        # Run entity extraction
        entities = self.entity("Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg")
        self.assertEqual([e[0] for e in entities], ["Canada", "Manhattan"])

    def testEntityFlatten(self):
        """
        Test entity with flattened output
        """

        # Test flatten
        entities = self.entity("Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", flatten=True)
        self.assertEqual(entities, ["Canada", "Manhattan"])

        # Test flatten with join
        entities = self.entity(
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", flatten=True, join=True
        )
        self.assertEqual(entities, "Canada Manhattan")

    def testEntityTypes(self):
        """
        Test entity type filtering
        """

        # Run entity extraction
        entities = self.entity("Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg", labels=["PER"])
        self.assertFalse(entities)

    def testEntityKwargs(self):
        """
        Test entity with **kwargs propagation
        """

        # Use batch_size kwarg (forwarded to HF pipeline)
        entities = self.entity(
            ["Canada's last fully intact ice shelf has suddenly collapsed", "Beijing mobilises invasion craft"],
            batch_size=2,
        )
        self.assertEqual(len(entities), 2)
        # Each result should be a list of entity tuples
        for result in entities:
            self.assertIsInstance(result, list)

    def testGliner(self):
        """
        Test entity pipeline with a GLiNER model
        """

        entity = Entity("neuml/gliner-bert-tiny")
        entities = entity("My name is John Smith.", flatten=True)
        self.assertEqual(entities, ["John Smith"])
