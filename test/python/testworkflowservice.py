"""
Service workflow task tests
"""

import unittest

from unittest.mock import patch

from requests.exceptions import Timeout

from txtai.workflow import ServiceTask


class TestWorkflowService(unittest.TestCase):
    """
    Service workflow task tests.
    """

    def testTimeout(self):
        """
        Test GET and POST forward configured timeouts and propagate timeout errors
        """

        for method in ("get", "post"):
            for options, timeout in (({}, 30), ({"timeout": 0.25}, 0.25), ({"timeout": None}, None)):
                with self.subTest(method=method, timeout=timeout):
                    task = ServiceTask(url="https://example.com/service", method=method, **options)
                    error = Timeout("request timed out")

                    with patch(f"txtai.workflow.task.service.requests.{method}", side_effect=error) as request:
                        with self.assertRaises(Timeout) as raised:
                            task.request({"text": "example"})

                    self.assertIs(raised.exception, error)
                    self.assertEqual(request.call_args.kwargs["timeout"], timeout)
