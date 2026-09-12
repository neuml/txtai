"""
Tests for placeholder Agent error messaging (#1161)
"""

import unittest


class TestPlaceholderAgentErrorMessage(unittest.TestCase):
    """
    Test that placeholder.Agent surfaces the real import failure reason
    instead of always blaming a missing smolagents install.
    """

    def testDefaultMessageUnchanged(self):
        """
        When no underlying import error is set, message is unchanged (backward compatible)
        """
        from txtai.agent.placeholder import Agent as PlaceholderAgent

        PlaceholderAgent._import_error = None
        try:
            PlaceholderAgent()
            self.fail("Expected ImportError")
        except ImportError as e:
            self.assertIn('smolagents is not available', str(e))
            self.assertNotIn('Underlying import error', str(e))
        finally:
            PlaceholderAgent._import_error = None

    def testUnderlyingImportErrorIncluded(self):
        """
        When an underlying import error is set (e.g. mcpadapt/mcp 2.0 incompatibility),
        the real cause is included in the message
        """
        from txtai.agent.placeholder import Agent as PlaceholderAgent

        PlaceholderAgent._import_error = (
            "cannot import name 'streamablehttp_client' from 'mcp.client.streamable_http'"
        )
        try:
            PlaceholderAgent()
            self.fail("Expected ImportError")
        except ImportError as e:
            self.assertIn('smolagents is not available', str(e))
            self.assertIn('Underlying import error', str(e))
            self.assertIn('mcp.client.streamable_http', str(e))
        finally:
            PlaceholderAgent._import_error = None


if __name__ == '__main__':
    unittest.main()
