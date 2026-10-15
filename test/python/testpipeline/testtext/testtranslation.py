"""
Translation module tests
"""

import unittest
import time
from unittest.mock import patch

import requests

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

        # Preload list of models. Handle HF Hub errors.
        complete, wait = False, 1
        while not complete:
            try:
                cls.translate.lookup("en", "es")
                complete = True
            except requests.exceptions.HTTPError:
                # Exponential backoff
                time.sleep(wait)

                # Wait up to 16 seconds
                wait = min(wait * 2, 16)

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

    def testLongTranslation(self):
        """
        Test a translation longer than max tokenization length
        """

        text = "This is a test translation to Spanish. " * 100
        translation = self.translate(text, "es")

        # Validate translation text
        self.assertIsNotNone(translation)

    def testM2M100Translation(self):
        """
        Test a translation using M2M100 models
        """

        text = self.translate("This is a test translation to Croatian", "hr")

        # Validate translation text
        self.assertEqual(text, "Ovo je testni prijevod na hrvatski")

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

    def testShowmodelsChunked(self):
        """
        Test a long translation with showmodels flag. When text is chunked
        by the tokenizer, results should still be properly concatenated as
        a 3-tuple (translation, language, model) rather than a malformed tuple.
        """

        text = "This is a test translation to Spanish. " * 100
        result = self.translate(text, "es", showmodels=True)

        # Result should be a tuple of exactly 3 elements
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

        translation, language, modelpath = result

        # Translation should be a single string, not a nested tuple
        self.assertIsInstance(translation, str)
        self.assertIsNotNone(translation)
        self.assertGreater(len(translation), 0)

        # Language and model should be valid strings
        self.assertEqual(language, "en")
        self.assertIsInstance(modelpath, str)

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


class TestTranslationSameLanguage(unittest.TestCase):
    """
    Model-free same-language translation tests.
    """

    def testSameLanguage(self):
        """
        Test scalar and batched inputs with and without model metadata.
        """

        translate = Translation(gpu=False, findmodels=False, batch=2)
        texts = ["hello", "hi", "a", "", "  hello  ", "This is a long text. " * 100]

        with patch.object(translate, "lookup") as lookup:
            for showmodels in (False, True):
                expected = [(text.strip(), "en", None) if showmodels else text.strip() for text in texts]
                with self.subTest(showmodels=showmodels):
                    self.assertEqual(translate(texts, source="en", target="en", showmodels=showmodels), expected)
                    self.assertEqual(translate([], source="en", target="en", showmodels=showmodels), [])
                for text, result in zip(texts, expected):
                    with self.subTest(text=text, showmodels=showmodels):
                        self.assertEqual(translate(text, source="en", target="en", showmodels=showmodels), result)

            lookup.assert_not_called()
