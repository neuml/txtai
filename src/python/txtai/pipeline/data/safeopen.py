"""
SafeOpen module
"""

import os
import tempfile

from urllib.parse import urlparse

from .urlretrieve import URLRetrieve


class SafeOpen:
    """
    Safely open a local file or URL. When `safeopen` is enabled, the following rules are applied.

      - Local files must be in the safeopen directory (defaults to temp dir)
      - URLs must be public URLs (also has checks for redirects and DNS rebind attempts)
    """

    def __init__(self, headers=None, safeopen=False, allowurl=True):
        """
        Creates a new safeopen instance.

        Args:
            safeopen: if safe validation checks should be enabled
            allowurl: if urls should be allowed
        """

        # HTTP headers
        self.headers = headers if headers else {}

        # Safe open mode. When set only local temp urls (or a specified directory) and non-private URLs are supported
        self.safeopen = os.path.realpath(tempfile.gettempdir() if isinstance(safeopen, bool) else safeopen) if safeopen else safeopen

        # URL retriever
        self.urlretrieve = URLRetrieve(self.headers, self.safeopen)

        # Enable URL validation
        self.allowurl = allowurl

    def valid(self, path):
        """
        Checks if path is a valid local file or web url. Returns path if valid along with a flag
        denoting if the path exists locally.

        Args:
            path: path to check

        Returns:
            (path, exists)
        """

        # Convert file urls to local paths
        path = path.replace("file://", "")

        # Check if this is a local file path or local file url
        exists = os.path.exists(path)

        # Safe open validation
        if not self.safecheck(path):
            path = path if urlparse(path).scheme else os.path.realpath(path)
            raise IOError(f"Safeopen URL validation failed: path={path}, safeopen={self.safeopen}")

        # Consider local files and HTTP urls valid
        return (path if exists or urlparse(path).scheme in ("http", "https") else None, exists)

    def retrieve(self, url):
        """
        Retrieves content from url.

        Args:
            url: input url

        Returns:
            data
        """

        # Local file
        if os.path.exists(url):
            with open(url, "rb") as f:
                return f.read()

        # Remote file
        return self.urlretrieve(url)

    def safecheck(self, url):
        """
        Safe open url validation. Validates a local path is within the safeopen directory and
        that URLs are a public HTTP(s) URLs.

        Args:
            url: input url

        Returns:
            True if url is valid, false otherwise
        """

        # Default to allow all urls when safe open is disabled
        valid = True

        if self.safeopen:
            if os.path.exists(url) or not self.allowurl:
                # Validate local file is in safe path
                path = os.path.realpath(url)
                prefix = os.path.commonpath([self.safeopen, path])
                valid = prefix == self.safeopen
            else:
                # URL validation
                valid = url.lower().startswith("http") and not self.urlretrieve.isprivateurl(url)

        return valid
