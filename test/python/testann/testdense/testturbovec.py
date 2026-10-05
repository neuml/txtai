"""
TurboVec ANN module tests
"""

from .base import DenseTest


class TestTurboVec(DenseTest):
    """
    TurboVec ANN tests.
    """

    def testTurboVec(self):
        """
        Test turbovec backend
        """

        self.runTests("turbovec")
