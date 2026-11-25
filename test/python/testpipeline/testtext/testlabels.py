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

    def testLabelFixedDefaultLabel2Id(self):
        """
        Test labels with a fixed label text classification model that has label names in id2label but a default label2id
        """

        labels = Labels(dynamic=False)
        text = "This is the best sentence ever"

        # Reset label2id to default names, as found in some model configs
        config = labels.pipeline.model.config
        config.label2id = {f"LABEL_{uid}": uid for uid in config.id2label}

        # Verify output label ids resolve using id2label
        for uid, name in config.id2label.items():
            self.assertEqual([x for x, _ in labels(text, labels=[name])], [uid])
            self.assertEqual(labels(text, labels=[name], flatten=True), [name])

        self.assertEqual(sorted(x for x, _ in labels(text)), sorted(config.id2label))

    def testLabelFixedFilter(self):
        """
        Test filtering labels with a fixed label text classification model. Covers every supported label form with every output format.
        """

        labels = Labels(dynamic=False)
        text = "This is the best sentence ever"
        config = labels.pipeline.model.config

        for uid, name in config.id2label.items():
            # Supported label forms: name (any case), integer id, id as string
            for label in [name, name.lower(), name.upper(), name[:1].lower() + name[1:].upper(), uid, str(uid)]:
                with self.subTest(label=label):
                    # (id, score) output
                    self.assertEqual([x for x, _ in labels(text, labels=[label])], [uid])
                    self.assertEqual([x for x, _ in labels(text, labels=[label], flatten=False)], [uid])

                    # Flattened output, top label and thresholds
                    self.assertEqual(labels(text, labels=[label], flatten=True), [name])
                    self.assertEqual(labels(text, labels=[label], flatten=0.0), [name])
                    self.assertEqual(labels(text, labels=[label], flatten=1.1), [])

                    # List input
                    self.assertEqual([[x for x, _ in row] for row in labels([text, text], labels=[label])], [[uid], [uid]])
                    self.assertEqual(labels([text, text], labels=[label], flatten=True), [[name], [name]])

        # Labels that don't match a model label are ignored
        unknown = max(config.id2label) + 1
        for label in ["unknown", unknown, str(unknown), True]:
            with self.subTest(label=label):
                self.assertEqual(labels(text, labels=[label]), [])
                self.assertEqual(labels(text, labels=[label], flatten=True), [])
                self.assertEqual(labels(text, labels=[label], flatten=0.0), [])

        # No filter returns all labels
        for value in [None, []]:
            with self.subTest(labels=value):
                self.assertEqual(len(labels(text, labels=value)), len(labels.labels()))
                self.assertEqual(labels(text, labels=value, flatten=True), labels(text, flatten=True))

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

    def testLabelFlatten(self):
        """
        Test labels with single text input, flattened to top text labels
        """

        self.assertEqual(self.labels("This is the best sentence ever", ["positive", "negative"], flatten=True)[0], "positive")

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
