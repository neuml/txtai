"""
URLRetrieve module
"""

import contextlib
import ipaddress
import socket

from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlparse, splitport
from urllib.request import (
    HTTPDefaultErrorHandler,
    HTTPErrorProcessor,
    HTTPHandler,
    HTTPSHandler,
    HTTPRedirectHandler,
    OpenerDirector,
    Request,
    UnknownHandler,
)

from ..base import Pipeline


class URLRetrieve(Pipeline):
    """
    Retrieves content from HTTP(s) URLs.
    """

    def __init__(self, headers=None, safeopen=False):
        """
        Creates a new URLRetrieve pipeline.

        Args:
            headers: http headers
            safeopen: if safe validation checks should be enabled
        """

        # HTTP headers
        self.headers = headers if headers else {}

        # Safeopen mode
        self.safeopen = safeopen

        # Create a blank opener
        self.opener = OpenerDirector()

        # Register handlers
        for handler in [
            UnknownHandler(),
            HTTPDefaultErrorHandler(),
            HTTPErrorProcessor(),
            SafeHTTPHandler(self),
            SafeHTTPSHandler(self),
            SafeRedirectHandler(self),
        ]:
            self.opener.add_handler(handler)

    def __call__(self, url):
        """
        Retrieves content from url.

        Args:
            url: input url

        Returns:
            data
        """

        with contextlib.closing(self.opener.open(Request(url, headers=self.headers))) as connection:
            return connection.read()

    def isprivateurl(self, url):
        """
        Checks if URL refers to a private/internal IP address.

        Args:
            url: input url

        Returns:
            True if this is a private url, false otherwise
        """

        # Assume the url is private until proven otherwise
        private = True

        try:
            host = urlparse(url).hostname
            if host:
                _, private = self.resolveip(host)

        # pylint: disable=W0718
        except Exception:
            pass

        return private

    def resolveip(self, host):
        """
        Resolves the IP Address for host.

        Args:
            host: host name

        Returns:
            (ip address, True if this is a private ip)
        """

        # Resolve IP Address
        ip = socket.gethostbyname(host)

        # Check if it's private
        return (ip, not ipaddress.ip_address(ip).is_global)


class SafeHTTPConnection(HTTPConnection):
    """
    HTTPConnection that pins to a supplied IP.
    """

    def __init__(self, host, ip, **kwargs):
        super().__init__(host, **kwargs)
        self.ip = ip

    def connect(self):
        self.sock = socket.create_connection(
            (self.ip, self.port),
            self.timeout or 5,
            self.source_address,
        )


class SafeHTTPSConnection(HTTPSConnection):
    """
    HTTPSConnection that pins to a supplied IP.
    """

    def __init__(self, host, ip, **kwargs):
        super().__init__(host, **kwargs)
        self.ip = ip

    def connect(self):
        raw = socket.create_connection(
            (self.ip, self.port),
            self.timeout or 5,
            self.source_address,
        )

        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


class SafeHTTPHandler(HTTPHandler):
    """
    SafeHTTPHandler that validates and pins a connection to an IP.
    """

    def __init__(self, instance):
        """
        Stores the current instance.

        Args:
            instance: validation instance
        """

        # Call parent constructor
        super().__init__()

        self.instance = instance

    def http_open(self, req):
        host = splitport(req.host)[0]

        # Validate and pin the IP Address
        ip, private = self.instance.resolveip(host)
        if self.instance.safeopen and private:
            raise IOError(f"Safeopen URL validation failed: host={host}")

        return self.do_open(
            lambda host, **kwargs: SafeHTTPConnection(host, ip, **kwargs),
            req,
        )


class SafeHTTPSHandler(HTTPSHandler):
    """
    SafeHTTPSHandler that validates and pins a connection to an IP.
    """

    def __init__(self, instance):
        """
        Stores the current instance.

        Args:
            instance: validation instance
        """

        # Call parent constructor
        super().__init__()

        self.instance = instance

    def https_open(self, req):
        host = splitport(req.host)[0]

        # Validate and pin the IP Address
        ip, private = self.instance.resolveip(host)
        if self.instance.safeopen and private:
            raise IOError(f"Safeopen URL validation failed: host={host}")

        return self.do_open(
            lambda host, **kwargs: SafeHTTPSConnection(host, ip, **kwargs),
            req,
        )


class SafeRedirectHandler(HTTPRedirectHandler):
    """
    Custom HTTPRedirectHandler that runs safe open url validation for each redirect hop.
    """

    def __init__(self, instance):
        """
        Stores the current instance.

        Args:
            instance: validation instance
        """

        self.instance = instance

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Validate redirects are safe
        if self.instance.safeopen and self.instance.isprivateurl(newurl):
            raise IOError(f"Safeopen URL validation failed: url={newurl}")

        # Pass through when URL is safe
        return super().redirect_request(req, fp, code, msg, headers, newurl)
