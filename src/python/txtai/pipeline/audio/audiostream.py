"""
AudioStream module
"""

from queue import Queue
from threading import Thread

import numpy as np

# Conditional import
try:
    import sounddevice as sd

    SOUNDDEVICE = True
except (ImportError, OSError):
    SOUNDDEVICE = False

from ..base import Pipeline


class AudioStream(Pipeline):
    """
    Threaded pipeline that streams audio segments to an output audio device. This pipeline is designed
    to run on local machines given that it requires access to write to an output device.
    """

    COMPLETE = 1

    def __init__(self, rate=22050):
        """
        Creates an AudioStream pipeline.

        Args:
            rate: sample rate of audio segments
        """

        if not SOUNDDEVICE:
            raise ImportError("SoundDevice library not installed or portaudio library not found")

        # Sampler rate
        self.rate = rate

        self.queue = Queue()
        self.thread = Thread(target=self.play)
        self.thread.start()

    def __call__(self, segment):
        # Convert single element to list
        segments = [segment] if isinstance(segment, np.ndarray) else segment

        for x in segments:
            self.queue.put(x)

        # Return single element if single element passed in
        return segments[0] if isinstance(segment, np.ndarray) else segments

    def wait(self):
        """
        Waits for all input audio segments to be played.
        """

        self.thread.join()

    def play(self):
        """
        Reads audio segments from queue. This method runs in a separate non-blocking thread.
        """

        audio = self.queue.get()
        while not isinstance(audio, int) or audio != AudioStream.COMPLETE:
            sd.play(audio, self.rate, blocking=True)
            audio = self.queue.get()
