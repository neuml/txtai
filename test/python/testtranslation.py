"""
Translation module tests
"""

import os
import unittest

from txtai.pipeline import Translation


class TestTranslation(unittest.TestCase):
    """
    Translation tests
    """

    def testMarianTranslation(self):
        """
        Tests a translation using Marian models
        """

        translate = Translation()

        # Validate translation from English - Spanish
        text = "This is a test translation into Spanish"
        translation = translate(text, "es")
        self.assertEqual(translation, "Esta es una traducción de prueba al español")

        # Validate no translation
        translation = translate("Esta es una traducción de prueba al español", "es")
        self.assertEqual(text, translation)

        # Validate long translation text
        text = "This is a test translation to Spanish. " * 100
        translation = translate(text, "es")
        self.assertIsNotNone(translation)

    @unittest.skipIf(os.name == "nt", "M2M100 skipped on Windows")
    def testM2M100Translation(self):
        """
        Tests a translation using M2M100 models
        """

        translate = Translation()

        text = translate("This is a test translation to Croatian", "hr")

        # Validate translation text
        self.assertEqual(text, "Ovo je testni prijevod na hrvatski")
