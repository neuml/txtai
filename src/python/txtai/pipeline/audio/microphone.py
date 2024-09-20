"""
Microphone module
"""

import numpy as np

# Conditional import
try:
    import speech_recognition as sr
    import webrtcvad

    PYAUDIO = True
except ImportError:
    PYAUDIO = False

from ..base import Pipeline


class Microphone(Pipeline):
    """
    Reads input audio from a microphone device. This pipeline is designed to run on local machines given
    that it requires access to read from an input device.
    """

    def __init__(self, rate=16000, vadmode=1, vadframe=30, vadthreshold=0.6):
        """
        Creates a new Microphone pipeline.

        Args:
            rate: sample rate to record audio in, defaults to 16 kHz
            vadmode: aggressiveness of the voice activity detector, defaults to 1
            vadframe: voice activity detector frame size in ms, defaults to 30
            vadthreshold: percentage of frames (0.0 - 1.0) that must be voice to be considered speech, defaults to 0.6
        """

        if not PYAUDIO:
            raise ImportError('Microphone pipeline is not available - install "pipeline" extra to enable')

        # Voice activity detector
        self.vad = webrtcvad.Vad(vadmode)
        self.vadframe = vadframe
        self.vadthreshold = vadthreshold

        # Sample rate
        self.rate = rate

        # Speech recognition config
        self.recognizer = sr.Recognizer()

    def __call__(self, device=None):
        # Read from microphone
        with sr.Microphone(sample_rate=self.rate) as source:
            # Calibrate microphone
            self.recognizer.adjust_for_ambient_noise(source)

            # Wait for speech
            audio = None
            while audio is None:
                audio = self.listen(source)

            # Return single element if single element passed in
            return (audio, self.rate) if device is None or not isinstance(device, list) else [(audio, self.rate)]

    def listen(self, source):
        """
        Listens for audio from source. Returns audio if it passes the voice
        activity detector.

        Args:
            source: microphone source

        Returns:
            audio if present, else None
        """

        audio = self.recognizer.listen(source)
        if self.detect(audio.frame_data, audio.sample_rate):
            # Convert to WAV
            data = audio.get_wav_data()

            # Convert to float32
            s16 = np.frombuffer(data, dtype=np.int16, count=len(data) // 2, offset=0)
            return s16.astype(np.float32, order="C") / 32768

        return None

    def detect(self, audio, rate):
        """
        Voice activity detector.

        Args:
            audio: input waveform data
            rate: sample rate

        Returns:
            True if the number of audio frames with audio pass vadthreshold, False otherwise
        """

        n = int(rate * (self.vadframe / 1000.0) * 2)
        offset = 0

        detects = []
        while offset + n < len(audio):
            detects.append(1 if self.vad.is_speech(audio[offset : offset + n], rate) else 0)
            offset += n

        return sum(detects) / len(detects) >= self.vadthreshold if detects else 0
