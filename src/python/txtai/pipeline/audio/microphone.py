"""
Microphone module
"""

import logging

import numpy as np

# Conditional import
try:
    import sounddevice as sd
    import webrtcvad

    from scipy.signal import butter, sosfilt

    from .signal import Signal, SCIPY

    MICROPHONE = SCIPY
except (ImportError, OSError):
    MICROPHONE = False

from ..base import Pipeline

# Logging configuration
logger = logging.getLogger(__name__)


class Microphone(Pipeline):
    """
    Reads input speech from a microphone device. This pipeline is designed to run on local machines given
    that it requires access to read from an input device.
    """

    def __init__(self, rate=16000, vadmode=3, vadframe=20, vadthreshold=0.6, voicestart=300, voiceend=3400, active=5, pause=8):
        """
        Creates a new Microphone pipeline.

        Args:
            rate: sample rate to record audio in, defaults to 16000 (16 kHz)
            vadmode: aggressiveness of the voice activity detector (1 - 3), defaults to 3, which is the most aggressive filter
            vadframe: voice activity detector frame size in ms, defaults to 20
            vadthreshold: percentage of frames (0.0 - 1.0) that must be voice to be considered speech, defaults to 0.6
            voicestart: starting frequency to use for voice filtering, defaults to 300
            voiceend: ending frequency to use for voice filtering, defaults to 3400
            active: minimum number of active speech chunks to require before considering this speech, defaults to 5
            pause: number of non-speech chunks to keep before considering speech complete, defaults to 8
        """

        if not MICROPHONE:
            raise ImportError(
                (
                    'Microphone pipeline is not available - install "pipeline" extra to enable. '
                    "Also check that the portaudio system library is available."
                )
            )

        # Sample rate
        self.rate = rate

        # Voice activity detector
        self.vad = webrtcvad.Vad(vadmode)
        self.vadframe = vadframe
        self.vadthreshold = vadthreshold

        # Voice spectrum
        self.voicestart = voicestart
        self.voiceend = voiceend

        # Audio chunks counts
        self.active = active
        self.pause = pause

    def __call__(self, device=None):
        """
        Reads audio from an input device.

        Args:
            device: optional input device id, otherwise uses system default

        Returns:
            list of (audio, sample rate)
        """

        # Listen for audio
        audio = self.listen(device[0] if isinstance(device, list) else device)

        # Return single element if single element passed in
        return (audio, self.rate) if device is None or not isinstance(device, list) else [(audio, self.rate)]

    def listen(self, device):
        """
        Listens for speech. Detected speech is converted to 32-bit floats for compatibility with
        automatic speech recognition (ASR) pipelines.

        This method blocks until speech is detected.

        Args:
            device: input device

        Returns:
            audio
        """

        # Record in 100ms chunks
        chunksize = self.rate // 10

        # Open input stream
        stream = sd.RawInputStream(device=device, samplerate=self.rate, channels=1, blocksize=chunksize, dtype=np.int16)

        # Start the input stream
        stream.start()

        record, speech, nospeech, chunks = True, 0, 0, []
        while record:
            # Read chunk
            chunk, _ = stream.read(chunksize)

            # Detect speech using WebRTC VAD for audio chunk
            detect = self.detect(chunk)
            speech = speech + 1 if detect else speech
            nospeech = 0 if detect else nospeech + 1

            # Save chunk, if this is an active stream
            if speech:
                chunks.append(chunk)

                # Pause limit has been reached, check if this audio should be accepted
                if nospeech >= self.pause:
                    logger.debug("Audio detected and being analyzed")
                    if speech >= self.active and self.isspeech(chunks[:-nospeech]):
                        # Disable recording
                        record = False
                    else:
                        # Reset parameters and keep recording
                        logger.debug("Speech not detected")
                        speech, nospeech, chunks = 0, 0, []

        # Stop the input stream
        stream.stop()

        # Convert to float32 and return
        audio = np.frombuffer(b"".join(chunks), np.int16)
        return Signal.float32(audio)

    def isspeech(self, chunks):
        """
        Runs an ensemble of Voice Activity Detection (VAD) methods. Returns true if speech is
        detected in the input audio chunks.

        Args:
            chunks: input audio chunks as byte buffers

        Returns:
            True if speech is detected, False otherwise
        """

        # Convert to NumPy array for processing
        audio = np.frombuffer(b"".join(chunks), dtype=np.int16)

        # Ensemble of:
        #  - WebRTC VAD with a human voice range butterworth bandpass filter applied to the signal
        #  - FFT applied to detect the energy ratio for human voice range vs total range
        return self.detectband(audio) and self.detectenergy(audio)

    def detect(self, buffer):
        """
        Detect speech using the WebRTC Voice Activity Detector (VAD).

        Args:
            buffer: input audio buffer frame as bytes

        Returns:
            True if the number of audio frames with audio pass vadthreshold, False otherwise
        """

        n = int(self.rate * (self.vadframe / 1000.0) * 2)
        offset = 0

        detects = []
        while offset + n <= len(buffer):
            detects.append(1 if self.vad.is_speech(buffer[offset : offset + n], self.rate) else 0)
            offset += n

        # Calculate detection ratio and return
        ratio = sum(detects) / len(detects) if detects else 0
        if ratio > 0:
            logger.debug("DETECT %.4f", ratio)

        return ratio >= self.vadthreshold

    def detectband(self, audio):
        """
        Detects speech using audio data filtered through a butterworth band filter
        with the human voice range.

        Args:
            audio: input audio data as an NumPy array

        Returns:
            True if speech is detected, False otherwise
        """

        # Upsample to float32
        audio = Signal.float32(audio)

        # Human voice frequency range
        low = self.voicestart / (0.5 * self.rate)
        high = self.voiceend / (0.5 * self.rate)

        # Low and high pass filter using human voice range
        sos = butter(5, Wn=[low, high], btype="band", output="sos")
        audio = sosfilt(sos, audio)

        # Scale back to int16
        audio = Signal.int16(audio)

        # Pass filtered signal to WebRTC VAD
        return self.detect(audio.tobytes())

    def detectenergy(self, audio):
        """
        Detects speech by comparing the signal energy of the human voice range
        to the overall signal energy.

        Args:
            audio: input audio data as an NumPy array

        Returns:
            True if speech is detected, False otherwise
        """

        # Calculate signal energy
        energyfreq = Signal.energy(audio, self.rate)

        # Sum speech energy
        speechenergy = 0
        for f, e in energyfreq.items():
            if self.voicestart <= f <= self.voiceend:
                speechenergy += e

        # Calculate ratio of speech energy to total energy and return
        ratio = speechenergy / sum(energyfreq.values())
        logger.debug("SPEECH %.4f", ratio)
        return ratio >= self.vadthreshold
