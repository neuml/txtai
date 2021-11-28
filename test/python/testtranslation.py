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

    def testLongTranslation(self):
        """
        Tests a translation longer than max tokenization length.
        """

        translate = Translation()

        text = "This is a test translation to Spanish. " * 100
        translation = translate(text, "es")

        # Validate translation text
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

    def testMarianTranslation(self):
        """
        Tests a translation using Marian models
        """

        translate = Translation()

        text = "This is a test translation into Spanish"
        translation = translate(text, "es")

        # Validate translation text
        self.assertEqual(translation, "Esta es una traducción de prueba al español")

        # Validate translation back
        translation = translate(translation, "en")
        self.assertEqual(translation, text)

    def testNoLang(self):
        """
        Test no matching language id.
        """

        translate = Translation()
        self.assertIsNone(translate.langid([], "zz"))

    def testNoModel(self):
        """
        Tests no known available model found.
        """

        translate = Translation()
        self.assertEqual(translate.modelpath("zz", "en"), "Helsinki-NLP/opus-mt-mul-en")

    def testNoTranslation(self):
        """
        Tests translation skipped when text already in destination language
        """

        translate = Translation()

        text = "This is a test translation to English"
        translation = translate(text, "en")

        # Validate no translation
        self.assertEqual(text, translation)
