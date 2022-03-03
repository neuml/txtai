"""
Transcription module
"""

try:
    import soundfile as sf

    SOUNDFILE = True
except (ImportError, OSError):
    SOUNDFILE = False

from transformers import AutoModelForCTC, Wav2Vec2Processor

from ..hfmodel import HFModel


class Transcription(HFModel):
    """
    Transcribes audio files to text.
    """

    def __init__(self, path="facebook/wav2vec2-base-960h", quantize=False, gpu=True, batch=64):
        """
        Constructs a new transcription pipeline.

        Args:
            path: optional path to model, accepts Hugging Face model hub id or local path,
                  uses default model for task if not provided
            quantize: if model should be quantized, defaults to False
            gpu: True/False if GPU should be enabled, also supports a GPU device id
            batch: batch size used to incrementally process content
        """

        # Call parent constructor
        super().__init__(path, quantize, gpu, batch)

        if not SOUNDFILE:
            raise ImportError("SoundFile library not installed or libsndfile not found")

        # load model and processor
        self.model = AutoModelForCTC.from_pretrained(self.path)
        self.processor = Wav2Vec2Processor.from_pretrained(self.path)

        # Move model to device
        self.model.to(self.device)

    def __call__(self, files):
        """
        Transcribes audio files to text.

        This method supports files as a string or a list. If the input is a string,
        the return type is string. If text is a list, the return type is a list.

        Args:
            files: text|list

        Returns:
            list of transcribed text
        """

        values = [files] if not isinstance(files, list) else files

        # Parse audio files
        speech = [sf.read(f) for f in values]

        # Get unique list of sampling rates
        unique = set(s[1] for s in speech)

        results = {}
        for sampling in unique:
            # Get inputs for current sampling rate
            inputs = [(x, s[0]) for x, s in enumerate(speech) if s[1] == sampling]

            # Transcribe in batches
            outputs = []
            for chunk in self.batch([s for _, s in inputs], self.batchsize):
                outputs.extend(self.transcribe(chunk, sampling))

            # Store output value
            for y, (x, _) in enumerate(inputs):
                results[x] = outputs[y].capitalize()

        # Return results in same order as input
        results = [results[x] for x in sorted(results)]
        return results[0] if isinstance(files, str) else results

    def transcribe(self, speech, sampling):
        """
        Transcribes audio to text.

        Args:
            speech: list of audio
            sampling: sampling rate

        Returns:
            list of transcribed text
        """

        with self.context():
            # Convert samples to features
            inputs = self.processor(speech, sampling_rate=sampling, padding=True, return_tensors=self.tensortype()).input_values

            # Place inputs on tensor device
            inputs = inputs.to(self.device)

            # Retrieve logits
            logits = self.model(inputs).logits

            # Decode argmax
            ids = self.argmax(logits, dimension=-1)

            return self.processor.batch_decode(ids)
