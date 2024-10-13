"""
AudioStream module
"""

from queue import Queue
from threading import Thread

# Conditional import
try:
    import sounddevice as sd

    from .signal import Signal, SCIPY

    AUDIOSTREAM = SCIPY
except (ImportError, OSError):
    AUDIOSTREAM = False

from ..base import Pipeline


class AudioStream(Pipeline):
    """
    Threaded pipeline that streams audio segments to an output audio device. This pipeline is designed
    to run on local machines given that it requires access to write to an output device.
    """

    # End of stream message
    COMPLETE = (1, None)

    def __init__(self, rate=None):
        """
        Creates an AudioStream pipeline.

        Args:
            rate: optional target sample rate, otherwise uses input target rate with each audio segment
        """

        if not AUDIOSTREAM:
            raise ImportError(
                (
                    'AudioStream pipeline is not available - install "pipeline" extra to enable. '
                    "Also check that the portaudio system library is available."
                )
            )

        # Target sample rate
        self.rate = rate

        self.queue = Queue()
        self.thread = Thread(target=self.play)
        self.thread.start()

    def __call__(self, segment):
        """
        Queues audio segments for the audio player.

        Args:
            segment: (audio, sample rate)|list

        Returns:
            segment
        """

        # Convert single element to list
        segments = [segment] if isinstance(segment, tuple) else segment

        for x in segments:
            self.queue.put(x)

        # Return single element if single element passed in
        return segments[0] if isinstance(segment, tuple) else segments

    def wait(self):
        """
        Waits for all input audio segments to be played.
        """

        self.thread.join()

    def play(self):
        """
        Reads audio segments from queue. This method runs in a separate non-blocking thread.
        """

        audio, rate = self.queue.get()
        while not isinstance(audio, int) or (audio, rate) != AudioStream.COMPLETE:
            # Resample to target sample rate, if necessary
            audio, rate = (Signal.resample(audio, rate, self.rate), self.rate) if self.rate else (audio, rate)

            # Play audio segment
            sd.play(audio, rate, blocking=True)

            # Get next segment
            audio, rate = self.queue.get()
