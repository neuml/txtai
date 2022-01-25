"""
Entity module tests
"""

import unittest

from txtai.pipeline import Entity


class TestEntity(unittest.TestCase):
    """
    Entity tests.
    """

    def testEntity(self):
        """
        Test entity
        """

        # Run entity extraction
        entity = Entity("dslim/bert-base-NER")
        entities = entity("Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg")

        self.assertEqual([e[0] for e in entities], ["Canada", "Manhattan"])
