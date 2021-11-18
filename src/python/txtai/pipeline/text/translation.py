"""
Translation module
"""

# Conditional import
try:
    import fasttext

    FASTTEXT = True
except ImportError:
    FASTTEXT = False

from huggingface_hub.hf_api import HfApi
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MarianMTModel, MarianTokenizer
from transformers.file_utils import cached_path

from ..hfmodel import HFModel


class Translation(HFModel):
    """
    Translates text from source language into target language.
    """

    # Default language detection model
    DEFAULT_LANG_DETECT = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"

    def __init__(self, path="facebook/m2m100_418M", quantize=False, gpu=True, batch=64, langdetect=DEFAULT_LANG_DETECT):
        """
        Constructs a new language translation pipeline.

        Args:
            path: optional path to model, accepts Hugging Face model hub id or local path,
                  uses default model for task if not provided
            quantize: if model should be quantized, defaults to False
            gpu: True/False if GPU should be enabled, also supports a GPU device id
            batch: batch size used to incrementally process content
            langdetect: path to language detection model, uses a default path if not provided
        """

        # Call parent constructor
        super().__init__(path, quantize, gpu, batch)

        # Language detection
        self.detector = None
        self.langdetect = langdetect

        # Language models
        self.models = {}
        self.ids = self.available()

    def __call__(self, texts, target="en", source=None):
        """
        Translates text from source language into target language.

        This method supports texts as a string or a list. If the input is a string,
        the return type is string. If text is a list, the return type is a list.

        Args:
            texts: text|list
            target: target language code, defaults to "en"
            source: source language code, detects language if not provided

        Returns:
            list of translated text
        """

        values = [texts] if not isinstance(texts, list) else texts

        # Detect source languages
        languages = self.detect(values) if not source else [source] * len(values)
        unique = set(languages)

        # Build list of (index, language, text)
        values = [(x, lang, values[x]) for x, lang in enumerate(languages)]

        results = {}
        for language in unique:
            # Get all text values for language
            inputs = [(x, text) for x, lang, text in values if lang == language]

            # Translate text in batches
            outputs = []
            for chunk in self.batch([text for _, text in inputs], self.batchsize):
                outputs.extend(self.translate(chunk, language, target))

            # Store output value
            for y, (x, _) in enumerate(inputs):
                results[x] = outputs[y]

        # Return results in same order as input
        results = [results[x] for x in sorted(results)]
        return results[0] if isinstance(texts, str) else results

    def available(self):
        """
        Runs a query to get a list of available language models from the Hugging Face API.

        Returns:
            list of available language name ids
        """

        return set(x.modelId for x in HfApi().list_models() if x.modelId.startswith("Helsinki-NLP"))

    def detect(self, texts):
        """
        Detects the language for each element in texts.

        Args:
            texts: list of text

        Returns:
            list of languages
        """

        if not FASTTEXT:
            raise ImportError('Language detection is not available - install "pipeline" extra to enable')

        if not self.detector:
            # Suppress unnecessary warning
            fasttext.FastText.eprint = lambda x: None

            # Load language detection model
            path = cached_path(self.langdetect)
            self.detector = fasttext.load_model(path)

        # Transform texts to format expected by language detection model
        texts = [x.lower().replace("\n", " ").replace("\r\n", " ") for x in texts]

        return [x[0].split("__")[-1] for x in self.detector.predict(texts)[0]]

    def translate(self, texts, source, target):
        """
        Translates text from source to target language.

        Args:
            texts: list of text
            source: source language code
            target: target language code

        Returns:
            list of translated text
        """

        # Return original if already in target language
        if source == target:
            return texts

        # Load model and tokenizer
        model, tokenizer = self.lookup(source, target)

        model.to(self.device)
        indices = None

        with self.context():
            if isinstance(model, M2M100ForConditionalGeneration):
                source = self.langid(tokenizer.lang_code_to_id, source)
                target = self.langid(tokenizer.lang_code_to_id, target)

                tokenizer.src_lang = source
                tokens, indices = self.tokenize(tokenizer, texts)

                translated = model.generate(**tokens, forced_bos_token_id=tokenizer.lang_code_to_id[target])
            else:
                tokens, indices = self.tokenize(tokenizer, texts)
                translated = model.generate(**tokens)

        # Decode translations
        translated = tokenizer.batch_decode(translated, skip_special_tokens=True)

        # Combine translations - handle splits on large text from tokenizer
        results, last = [], -1
        for x, i in enumerate(indices):
            if i == last:
                results[-1] += translated[x]
            else:
                results.append(translated[x])

            last = i

        return results

    def lookup(self, source, target):
        """
        Retrieves a translation model for source->target language. This method caches each model loaded.

        Args:
            source: source language code
            target: target language code

        Returns:
            (model, tokenizer)
        """

        # Determine best translation model to use, load if necessary and return
        path = self.modelpath(source, target)
        if path not in self.models:
            self.models[path] = self.load(path)

        return self.models[path]

    def modelpath(self, source, target):
        """
        Derives a translation model path given source and target languages.

        Args:
            source: source language code
            target: target language code

        Returns:
            model path
        """

        # First try direct model
        template = "Helsinki-NLP/opus-mt-%s-%s"
        path = template % (source, target)
        if path in self.ids:
            return path

        # Use multi-language - english model
        if target == "en":
            return template % ("mul", target)

        # Default model if no suitable model found
        return self.path

    def load(self, path):
        """
        Loads a model specified by path.

        Args:
            path: model path

        Returns:
            (model, tokenizer)
        """

        if path.startswith("Helsinki-NLP"):
            model = MarianMTModel.from_pretrained(path)
            tokenizer = MarianTokenizer.from_pretrained(path)
        else:
            model = M2M100ForConditionalGeneration.from_pretrained(path)
            tokenizer = M2M100Tokenizer.from_pretrained(path)

        # Apply model initialization routines
        model = self.prepare(model)

        return (model, tokenizer)

    def langid(self, languages, target):
        """
        Searches a list of languages for a prefix match on target.

        Args:
            languages: list of languages
            target: target language code

        Returns:
            best match or None if no match found
        """

        for lang in languages:
            if lang.startswith(target):
                return lang

        return None
