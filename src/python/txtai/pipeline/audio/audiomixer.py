"""
AudioMixer module
"""

from ..base import Pipeline
from .signal import Signal, SCIPY


class AudioMixer(Pipeline):
    """
    Mixes multiple audio streams into a single stream.
    """

    def __init__(self, rate=None):
        """
        Creates an AudioMixer pipeline.

        Args:
            rate: optional target sample rate, otherwise uses input target rate with each audio segment
        """

        if not SCIPY:
            raise ImportError('AudioMixer pipeline is not available - install "pipeline" extra to enable.')

        # Target sample rate
        self.rate = rate

    def __call__(self, segment, scale1=1, scale2=1):
        """
        Mixes multiple audio streams into a single stream.

        Args:
            segment: ((audio1, sample rate), (audio2, sample rate))|list
            scale1: optional scaling factor for segment1
            scale2: optional scaling factor for segment2

        Returns:
            list of (audio, sample rate)
        """

        # Convert single element to list
        segments = [segment] if isinstance(segment, tuple) else segment

        results = []
        for segment1, segment2 in segments:
            audio1, rate1 = segment1
            audio2, rate2 = segment2

            # Resample audio, as necessary
            target = self.rate if self.rate else rate1
            audio1 = Signal.resample(audio1, rate1, target)
            audio2 = Signal.resample(audio2, rate2, target)

            # Mix audio into single segment
            results.append((Signal.mix(audio1, audio2, scale1, scale2), target))

        # Return single element if single element passed in
        return results[0] if isinstance(segment, tuple) else results
