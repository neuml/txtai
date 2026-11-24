"""
Labels module tests
"""

import unittest

from txtai.pipeline import Labels


class TestLabels(unittest.TestCase):
    """
    Labels tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create single labels instance.
        """

        cls.data = [
            "US tops 5 million confirmed virus cases",
            "Canada's last fully intact ice shelf has suddenly collapsed, forming a Manhattan-sized iceberg",
            "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
            "The National Park Service warns against sacrificing slower friends in a bear attack",
            "Maine man wins $1M from $25 lottery ticket",
            "Make huge profits without work, earn up to $100,000 a day",
        ]

        cls.labels = Labels("prajjwal1/bert-medium-mnli")

    def testLabel(self):
        """
        Test labels with single text input
        """

        self.assertEqual(self.labels("This is the best sentence ever", ["positive", "negative"])[0][0], 0)

    def testLabelFlatten(self):
        """
        Test labels with single text input, flattened to top text labels
        """

        self.assertEqual(self.labels("This is the best sentence ever", ["positive", "negative"], flatten=True)[0], "positive")

    def testLabelBatch(self):
        """
        Test labels with multiple text inputs
        """

        results = [l[0][0] for l in self.labels(["This is the best sentence ever", "This is terrible"], ["positive", "negative"])]
        self.assertEqual(results, [0, 1])

    def testLabelBatchFlatten(self):
        """
        Test labels with multiple text inputs, flattened to top text labels
        """

        results = [l[0] for l in self.labels(["This is the best sentence ever", "This is terrible"], ["positive", "negative"], flatten=True)]
        self.assertEqual(results, ["positive", "negative"])

    def testLabelZeroThreshold(self):
        """
        Test numeric zero flattening thresholds with dynamic and fixed labels
        """

        text = "This is the best sentence ever"
        for classifier, candidates in ((self.labels, ["positive", "negative"]), (Labels(dynamic=False), ["POSITIVE", "NEGATIVE"])):
            with self.subTest(dynamic=classifier.dynamic):
                self.assertCountEqual(classifier(text, candidates, flatten=0.0), candidates)
                self.assertCountEqual(classifier([text], candidates, flatten=0.0)[0], candidates)

                # Boolean False still requests the original score tuples.
                self.assertEqual(classifier(text, candidates, flatten=False), classifier(text, candidates))

    def testLabelFixed(self):
        """
        Test labels with a fixed label text classification model
        """

        labels = Labels(dynamic=False)

        # Get index of "POSITIVE" label
        index = labels.labels().index("POSITIVE")

        # Verify results
        self.assertEqual(labels("This is the best sentence ever")[0][0], index)
        self.assertEqual(labels("This is the best sentence ever", multilabel=True)[0][0], index)

    def testLabelFixedFlatten(self):
        """
        Test labels with a fixed label text classification model, flattened to top text labels
        """

        labels = Labels(dynamic=False)

        # Verify results
        self.assertEqual(labels("This is the best sentence ever", flatten=True)[0], "POSITIVE")
        self.assertEqual(labels("This is the best sentence ever", multilabel=True, flatten=True)[0], "POSITIVE")

    def testLabelFixedLimit(self):
        """
        Test labels with a fixed label text classification model, limited to requested labels by name and by id
        """

        labels = Labels(dynamic=False)

        # Rotate the config label mappings so insertion order no longer matches label ids
        config = labels.pipeline.model.config
        ids = list(config.id2label)
        config.id2label = {x: config.id2label[x] for x in ids[1:] + ids[:1]}
        config.label2id = {label: x for x, label in config.id2label.items()}

        # Verify requested labels resolve to label ids, not positions in the config label mappings
        for label, uid in config.label2id.items():
            self.assertEqual(labels("This is the best sentence ever", labels=[label.lower()])[0][0], uid)
            self.assertEqual(labels("This is the best sentence ever", labels=[str(uid)])[0][0], uid)
            self.assertEqual(labels("This is the best sentence ever", labels=[uid])[0][0], uid)

    def testLabelFixedFlattenFilter(self):
        """
        Test labels with a fixed label text classification model, filtering flattened output
        """

        labels = Labels(dynamic=False)
        text = "This is the best sentence ever"
        config = labels.pipeline.model.config

        for uid, name in config.id2label.items():
            mixed = name[:1].lower() + name[1:].upper()

            self.assertEqual(labels(text, labels=[name.upper()], flatten=True)[0], name)
            self.assertEqual(labels(text, labels=[name.lower()], flatten=True)[0], name)
            self.assertEqual(labels(text, labels=[mixed], flatten=True)[0], name)
            self.assertEqual(labels(text, labels=[str(uid)], flatten=True)[0], name)
            self.assertEqual(labels(text, labels=[uid], flatten=True)[0], name)

            filtered = labels(text, labels=[uid], flatten=False)
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0][0], uid)

            self.assertEqual(labels(text, labels=[name], flatten=0.0), [name])
            self.assertEqual(labels(text, labels=[name], flatten=1.1), [])

        self.assertEqual(labels(text, labels=["unknown"], flatten=True), [])
        self.assertEqual(labels(text, labels=[str(max(config.id2label) + 1)], flatten=True), [])
        self.assertEqual(labels(text, labels=[str(max(config.id2label) + 1)], flatten=False), [])
        self.assertEqual(labels(text, labels=[max(config.id2label) + 1], flatten=False), [])
        self.assertEqual(labels(text, labels=[max(config.id2label) + 1], flatten=True), [])

        self.assertEqual(labels(text, labels=["unknown"], flatten=False), [])
        self.assertTrue(labels(text, labels=["unknown"], flatten=1.1) == [])

    def testLabelFixedWithoutFilter(self):
        """
        Test fixed label behavior when no label filter is provided
        """

        labels = Labels(dynamic=False)
        text = "This is the best sentence ever"

        self.assertTrue(labels(text, flatten=True))
        self.assertEqual(len(labels(text, flatten=False)), len(labels.labels()))
        self.assertEqual(labels(text, labels=None, flatten=False), labels(text, flatten=False))
