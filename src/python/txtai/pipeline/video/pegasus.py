"""
Pegasus module
"""

import os

# Conditional import
try:
    from twelvelabs import TwelveLabs
    from twelvelabs.types import VideoContext_AssetId, VideoContext_Url

    TWELVELABS = True
except ImportError:
    TWELVELABS = False

from ..base import Pipeline


class Pegasus(Pipeline):
    """
    Analyzes and generates text from videos using the TwelveLabs Pegasus model via the TwelveLabs API.

    Pegasus is a video understanding model. Given a video and a prompt, it can summarize, caption,
    answer questions about or extract structured information from the video. Videos are referenced by
    a public url or a TwelveLabs asset id (for previously uploaded videos).
    """

    def __init__(self, path=None, api_key=None, maxtokens=2048, **kwargs):
        """
        Creates a new Pegasus pipeline.

        Args:
            path: Pegasus model name, defaults to pegasus1.5
            api_key: TwelveLabs API key, defaults to the TWELVELABS_API_KEY environment variable
            maxtokens: default maximum number of tokens to generate per response
            kwargs: additional arguments passed to the TwelveLabs client constructor
        """

        if not TWELVELABS:
            raise ImportError('Pegasus pipeline is not available - install "pipeline" extra to enable')

        # Default model
        self.path = path if path else "pegasus1.5"
        self.maxtokens = maxtokens

        # Build API client. API key resolved from parameter or the TWELVELABS_API_KEY environment variable.
        self.client = TwelveLabs(api_key=api_key if api_key else os.environ.get("TWELVELABS_API_KEY"), **kwargs)

    def __call__(self, video, prompt, maxtokens=None, **kwargs):
        """
        Analyzes one or more videos with a prompt and generates text.

        This method supports a single video or a list of videos. If the input is a single video, the
        return type is a string. If the input is a list, a list of strings is returned.

        A video is referenced by a public url (str) or, for previously uploaded content, an asset id
        passed as {"asset_id": "..."}.

        Args:
            video: video url|asset id dict, or a list of either
            prompt: analysis prompt
            maxtokens: maximum number of tokens to generate, defaults to the pipeline default
            kwargs: additional arguments passed to the analyze API (e.g. temperature)

        Returns:
            generated text or list of generated text
        """

        # Detect single input
        values = [video] if not isinstance(video, list) else video

        # Analyze each video
        results = [self.analyze(x, prompt, maxtokens, **kwargs) for x in values]

        # Return single result for single input
        return results[0] if not isinstance(video, list) else results

    def analyze(self, video, prompt, maxtokens, **kwargs):
        """
        Runs analysis for a single video.

        Args:
            video: video url or asset id dict
            prompt: analysis prompt
            maxtokens: maximum number of tokens to generate
            kwargs: additional analyze API arguments

        Returns:
            generated text
        """

        # Build video context - asset id dict routes to an uploaded asset, otherwise treat as a url
        context = VideoContext_AssetId(asset_id=video["asset_id"]) if isinstance(video, dict) else VideoContext_Url(url=video)

        # Call the analyze API
        response = self.client.analyze(
            model_name=self.path, video=context, prompt=prompt, max_tokens=maxtokens if maxtokens else self.maxtokens, **kwargs
        )

        return response.data
