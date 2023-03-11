"""
Translation module tests
"""

import os
import unittest

from txtai.pipeline import Translation


class TestTranslation(unittest.TestCase):
    """
    Translation tests.
    """

    def testDetect(self):
        """
        Test language detection
        """
        translate = Translation()

        test = ["This is a test language detection."]
        language = translate.detect(test)

        self.assertListEqual(language, ["en"])

    def testDetectWithCustomFunc(self):
        """
        Test language detection with custom function
        """

        def dummy_func(text):
            return ["en" for x in text]

        translate = Translation(langdetect=dummy_func)

        test = ["This is a test language detection."]
        language = translate.detect(test)

        self.assertListEqual(language, ["en"])

    def testLongTranslation(self):
        """
        Test a translation longer than max tokenization length
        """

        translate = Translation()

        text = "This is a test translation to Spanish. " * 100
        translation = translate(text, "es")

        # Validate translation text
        self.assertIsNotNone(translation)

    @unittest.skipIf(os.name == "nt", "M2M100 skipped on Windows")
    def testM2M100Translation(self):
        """
        Test a translation using M2M100 models
        """

        translate = Translation()

        text = translate("This is a test translation to Croatian", "hr")

        # Validate translation text
        self.assertEqual(text, "Ovo je testni prijevod na hrvatski")

    def testMarianTranslation(self):
        """
        Test a translation using Marian models
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
        Test no matching language id
        """

        translate = Translation()
        self.assertIsNone(translate.langid([], "zz"))

    def testNoModel(self):
        """
        Test no known available model found
        """

        translate = Translation()
        self.assertEqual(translate.modelpath("zz", "en"), "Helsinki-NLP/opus-mt-mul-en")

    def testNoTranslation(self):
        """
        Test translation skipped when text already in destination language
        """

        translate = Translation()

        text = "This is a test translation to English"
        translation = translate(text, "en")

        # Validate no translation
        self.assertEqual(text, translation)

    def testTranslationWithShowmodels(self):
        """
        Test a translation using Marian models and showmodels flag to return
        model and language.
        """

        translate = Translation()

        text = "This is a test translation into Spanish"
        result = translate(text, "es", showmodels=True)

        translation, language, modelpath = result
        # Validate translation text
        self.assertEqual(translation, "Esta es una traducción de prueba al español")
        # Validate detected language
        self.assertEqual(language, "en")
        # Validate model
        self.assertEqual(modelpath, "Helsinki-NLP/opus-mt-en-es")

        # Validate translation back
        result = translate(translation, "en", showmodels=True)

        translation, language, modelpath = result
        self.assertEqual(translation, text)
        # Validate detected language
        self.assertEqual(language, "es")
        # Validate model
        self.assertEqual(modelpath, "Helsinki-NLP/opus-mt-es-en")
