"""
Transcription module
"""

try:
    import soundfile as sf

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

    def __call__(self, audio, rate=None):
        """
        Transcribes audio files or data to text.

        This method supports a single audio element or a list of audio. If the input is audio, the return
        type is a string. If text is a list, a list of strings is returned

        Args:
            audio: audio|list
            rate: sampling rate, only required with raw audio data

        Returns:
            list of transcribed text
        """

        # Convert single element to list
        values = [audio] if not isinstance(audio, list) else audio

        # Parse audio
        speech = self.parse(values, rate)

        # Apply transformation rules and store results
        results = []
        for result in self.pipeline(speech):
            # Trim whitespace
            text = result["text"].strip()

            # Convert all upper case strings to capitalized case
            text = text.capitalize() if text.isupper() else text

            # Store result
            results.append(text)

        # Return single element if single element passed in
        return results[0] if not isinstance(audio, list) else results

    def parse(self, audio, rate):
        """
        Parses audio to raw waveforms and sampling rates.

        Args:
            audio: audio|list
            rate: optional sampling rate

        Returns:
            List of dictionaries
        """

        speech = []
        for x in audio:
            if isinstance(x, str):
                # Read file
                raw, samplerate = sf.read(x)
            else:
                # Input is NumPy array
                raw, samplerate = x, rate

            speech.append({"raw": raw, "sampling_rate": samplerate})

        return speech
