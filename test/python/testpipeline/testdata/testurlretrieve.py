"""
URLRetrieve module tests
"""

import contextlib
import unittest

from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from urllib.request import build_opener

from txtai.pipeline import URLRetrieve
from txtai.pipeline.data.urlretrieve import SafeRedirectHandler


class RequestHandler(BaseHTTPRequestHandler):
    """
    Test HTTP handler.
    """

    def do_GET(self):
        """
        GET request handler.
        """

        if self.path == "/valid":
            redirect = "https://github.com/neuml/txtai"
        elif self.path == "/invalid":
            redirect = "http://127.0.0.1"
        else:
            redirect = None

        if redirect:
            self.send_response(301)
            self.send_header("Location", redirect)
            self.end_headers()
        else:
            response = "test".encode("utf-8")

            self.send_response(200)
            self.send_header("content-type", "text/plain")
            self.send_header("content-length", len(response))
            self.end_headers()

            self.wfile.write(response)
            self.wfile.flush()


class TestURLRetrieve(unittest.TestCase):
    """
    URLRetrieve tests.
    """

    @classmethod
    def setUpClass(cls):
        """
        Create mock http server
        """

        cls.httpd = HTTPServer(("127.0.0.1", 8006), RequestHandler)

        server = Thread(target=cls.httpd.serve_forever, daemon=True)
        server.start()

    @classmethod
    def tearDownClass(cls):
        """
        Shutdown mock http server.
        """

        cls.httpd.shutdown()

    def testRedirect(self):
        """
        Test redirects
        """

        urlretrieve = URLRetrieve(safeopen=True)

        # Test redirect handler
        opener = build_opener(SafeRedirectHandler(urlretrieve))

        # Test valid direct
        with contextlib.closing(opener.open("http://127.0.0.1:8006/valid")) as connection:
            self.assertTrue("txtai is an all-in-one AI framework" in str(connection.read()))

        # Test invalid redirect
        with self.assertRaises(IOError):
            contextlib.closing(opener.open("http://127.0.0.1:8006/invalid"))

    def testRetrieve(self):
        """
        Test retrieval
        """

        urlretrieve = URLRetrieve()
        data = urlretrieve("http://127.0.0.1:8006/data")
        self.assertEqual(data, b"test")

    def testSafeopen(self):
        """
        Test safeopen checks
        """

        urlretrieve = URLRetrieve(safeopen=True)

        # Verify that local ip addresses fail
        with self.assertRaises(IOError):
            urlretrieve("http://127.0.0.1")

        with self.assertRaises(IOError):
            urlretrieve("https://127.0.0.1")
