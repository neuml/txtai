"""
Zvec ANN module tests
"""

import numpy as np

from txtai.ann import ANNFactory

from .base import DenseTest


class TestZvec(DenseTest):
    """
    Zvec ANN tests.
    """

    def testZvec(self):
        """
        Test zvec backend
        """

        self.runTests("zvec")
        self.deletenumpy("zvec")

    def testZvecCustom(self):
        """
        Test zvec backend with custom settings
        """

        self.runTests("zvec", {"zvec": {"efconstruction": 100, "m": 16, "efsearch": 50}})

        # Test invalid file path handled
        with self.assertRaises(FileNotFoundError):
            ann = ANNFactory.create({"backend": "zvec"})
            ann.load("non-exist-path")

    def testZvecDatatype(self):
        """
        Test zvec backend vector data type setting
        """

        # fp32 is the default, fp16 is a supported alternative
        for datatype, expected in (("fp32", "VECTOR_FP32"), ("fp16", "VECTOR_FP16")):
            self.runTests("zvec", {"zvec": {"datatype": datatype}}, update=False)

            ann = ANNFactory.create({"backend": "zvec", "dimensions": 8, "zvec": {"datatype": datatype}})
            self.assertEqual(ann.datatype().name, expected)

        # int8 stores integer vectors
        data = np.random.randint(0, 100, (100, 8)).astype(np.int8)
        ann = ANNFactory.create({"backend": "zvec", "dimensions": data.shape[1], "zvec": {"datatype": "int8"}})
        ann.index(data)

        self.assertEqual(ann.count(), 100)
        self.assertEqual(len(ann.search(data[:1], 1)[0]), 1)

        # Unsupported values are rejected with the supported options
        ann = ANNFactory.create({"backend": "zvec", "dimensions": 8, "zvec": {"datatype": "fp64"}})
        with self.assertRaises(ValueError):
            ann.index(np.random.rand(4, 8).astype(np.float32))
