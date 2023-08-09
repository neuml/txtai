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

    @classmethod
    def setUpClass(cls):
        """
        Create single translation instance.
        """

        cls.translate = Translation()

    def testDetect(self):
        """
        Test language detection
        """

        test = ["This is a test language detection."]
        language = self.translate.detect(test)

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

    @unittest.skipIf(os.name == "nt", "testLongTranslation skipped on Windows")
    def testLongTranslation(self):
        """
        Test a translation longer than max tokenization length
        """

        text = "This is a test translation to Spanish. " * 100
        translation = self.translate(text, "es")

        # Validate translation text
        self.assertIsNotNone(translation)

    @unittest.skipIf(os.name == "nt", "M2M100 skipped on Windows")
    def testM2M100Translation(self):
        """
        Test a translation using M2M100 models
        """

        text = self.translate("This is a test translation to Croatian", "hr")

        # Validate translation text
        self.assertEqual(text, "Ovo je testni prijevod na hrvatski")

    @unittest.skipIf(os.name == "nt", "testMarianTranslation skipped on Windows")
    def testMarianTranslation(self):
        """
        Test a translation using Marian models
        """

        text = "This is a test translation into Spanish"
        translation = self.translate(text, "es")

        # Validate translation text
        self.assertEqual(translation, "Esta es una traducción de prueba al español")

        # Validate translation back
        translation = self.translate(translation, "en")
        self.assertEqual(translation, text)

    def testNoLang(self):
        """
        Test no matching language id
        """

        self.assertIsNone(self.translate.langid([], "zz"))

    def testNoModel(self):
        """
        Test no known available model found
        """

        self.assertEqual(self.translate.modelpath("zz", "en"), "Helsinki-NLP/opus-mt-mul-en")

    def testNoTranslation(self):
        """
        Test translation skipped when text already in destination language
        """

        text = "This is a test translation to English"
        translation = self.translate(text, "en")

        # Validate no translation
        self.assertEqual(text, translation)

    @unittest.skipIf(os.name == "nt", "testTranslationWithShowModels skipped on Windows")
    def testTranslationWithShowmodels(self):
        """
        Test a translation using Marian models and showmodels flag to return
        model and language.
        """

        text = "This is a test translation into Spanish"
        result = self.translate(text, "es", showmodels=True)

        translation, language, modelpath = result
        # Validate translation text
        self.assertEqual(translation, "Esta es una traducción de prueba al español")
        # Validate detected language
        self.assertEqual(language, "en")
        # Validate model
        self.assertEqual(modelpath, "Helsinki-NLP/opus-mt-en-es")

        # Validate translation back
        result = self.translate(translation, "en", showmodels=True)

        translation, language, modelpath = result
        self.assertEqual(translation, text)
        # Validate detected language
        self.assertEqual(language, "es")
        # Validate model
        self.assertEqual(modelpath, "Helsinki-NLP/opus-mt-es-en")
