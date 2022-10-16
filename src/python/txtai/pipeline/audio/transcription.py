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
    Transcribes audio files to text.
    """

    def __init__(self, path=None, quantize=False, gpu=True, model=None):
        if not SOUNDFILE:
            raise ImportError("SoundFile library not installed or libsndfile not found")

        # Call parent constructor
        super().__init__("automatic-speech-recognition", path, quantize, gpu, model)

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

        # Format inputs
        speech = [{"raw": s[0], "sampling_rate": s[1]} for s in speech]

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
        return results[0] if isinstance(files, str) else results
