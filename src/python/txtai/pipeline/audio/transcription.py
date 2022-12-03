"""
Transcription module
"""

try:
    import soundfile as sf

    from scipy import signal

    SOUNDFILE = True
except (ImportError, OSError):
    SOUNDFILE = False

from ..hfpipeline import HFPipeline


class Transcription(HFPipeline):
    """
    Transcribes audio files or data to text.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        if not SOUNDFILE:
            raise ImportError("SoundFile library not installed or libsndfile not found")

        # Call parent constructor
        super().__init__("automatic-speech-recognition", path, quantize, gpu, model)

    def __call__(self, audio, rate=None, chunk=10, join=True):
        """
        Transcribes audio files or data to text.

        This method supports a single audio element or a list of audio. If the input is audio, the return
        type is a string. If text is a list, a list of strings is returned

        Args:
            audio: audio|list
            rate: sample rate, only required with raw audio data
            chunk: process audio in chunk second sized segments
            join: if True (default), combine each chunk back together into a single text output.
                  When False, chunks are returned as a list of dicts, each having raw associated audio and
                  sample rate in addition to text

        Returns:
            list of transcribed text
        """

        # Convert single element to list
        values = [audio] if not isinstance(audio, list) else audio

        # Read input audio
        speech = self.read(values, rate)

        # Apply transformation rules and store results
        results = self.batchprocess(speech, chunk) if chunk and not join else self.process(speech, chunk)

        # Return single element if single element passed in
        return results[0] if not isinstance(audio, list) else results

    def read(self, audio, rate):
        """
        Read audio to raw waveforms and sample rates.

        Args:
            audio: audio|list
            rate: optional sample rate

        Returns:
            List of (audio data, sample rate)
        """

        speech = []
        for x in audio:
            if isinstance(x, str):
                # Read file
                raw, samplerate = sf.read(x)
            elif isinstance(x, tuple):
                # Input is NumPy array and sample rate
                raw, samplerate = x
            else:
                # Input is NumPy array
                raw, samplerate = x, rate

            speech.append((raw, samplerate))

        return speech

    def process(self, speech, chunk):
        """
        Standard processing loop. Runs a single pipeline call for all speech inputs along
        with the chunk size. Returns text for each input.

        Args:
            speech: list of (audio data, sample rate)
            chunk: split audio into chunk seconds sized segments for processing

        Returns:
            list of transcribed text
        """

        results = []
        for result in self.pipeline([self.convert(*x) for x in speech], chunk_length_s=chunk, ignore_warning=True):
            # Store result
            results.append(self.clean(result["text"]))

        return results

    def batchprocess(self, speech, chunk):
        """
        Batch processing loop. Runs a pipeline call per speech input. Each speech input is split
        into chunk duration segments. Each segment is individually transcribed and returned along with
        the raw wav snippets.

        Args:
            speech: list of (audio data, sample rate)
            chunk: split audio into chunk seconds sized segments for processing

        Returns:
            list of lists of dicts - each dict has text, raw wav data for text and sample rate
        """

        results = []

        # Process each element individually to get time-sliced chunks
        for raw, rate in speech:
            # Get segments for current speech entry
            segments = self.segments(raw, rate, chunk)

            # Process segments, store raw data before processing given pipeline modifies it
            sresults = []
            for x, result in enumerate(self.pipeline([self.convert(*x) for x in segments])):
                sresults.append({"text": self.clean(result["text"]), "raw": segments[x][0], "rate": segments[x][1]})

            results.append(sresults)

        return results

    def segments(self, raw, rate, chunk):
        """
        Builds chunk duration batches.

        Args:
            raw: raw audio data
            rate: sample rate
            chunk: chunk duration size
        """

        segments = []

        # Split into batches, use sample rate * chunk seconds
        for segment in self.batch(raw, rate * chunk):
            segments.append((segment, rate))

        return segments

    def convert(self, raw, rate):
        """
        Converts input audio to mono with a sample rate equal to the pipeline model's
        sample rate.

        Args:
            raw: raw audio data

        Returns:
            audio data ready for pipeline model
        """

        # Convert stereo to mono, if necessary
        raw = self.mono(raw)

        # Resample to model sample rate
        raw, rate = self.resample(raw, rate)

        return {"raw": raw, "sampling_rate": rate}

    def mono(self, raw):
        """
        Convert stereo to mono audio.

        Args:
            raw: raw audio data

        Returns:
            audio data with a single channel
        """

        return raw.mean(axis=1) if len(raw.shape) > 1 else raw

    def resample(self, raw, rate):
        """
        Resample raw audio if the sample rate doesn't match the sample rate required for this model.

        Args:
            raw: raw audio data
            rate: sample rate

        Returns:
            raw audio resampled if necessary or original raw audio
        """

        targetrate = self.pipeline.feature_extractor.sampling_rate
        if rate != targetrate:
            samples = round(len(raw) * float(targetrate) / rate)
            raw = signal.resample(raw, samples)

        return raw, targetrate

    def clean(self, text):
        """
        Applies text normalization rules.

        Args:
            text: input text

        Returns:
            clean text
        """

        # Trim whitespace
        text = text.strip()

        # Convert all upper case strings to capitalized case
        return text.capitalize() if text.isupper() else text
