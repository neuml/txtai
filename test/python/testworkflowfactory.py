"""
Workflow factory tests
"""

import copy
import unittest

from txtai.app import Application
from txtai.workflow import StreamTask, TaskFactory, WorkflowFactory


class TestWorkflowFactory(unittest.TestCase):
    """
    Workflow configuration reuse tests.
    """

    def testApplicationConfig(self):
        """
        Test repeated application construction preserves workflow configuration
        """

        config = Application.read(
            """
            workflow:
                template:
                    tasks:
                        - task: template
                          template: Hi {text}
                        - action: testworkflowfactory.upper
                          args: ["?"]
                stream:
                    stream:
                        task: stream
                        action: testapp.TestStream
                        batch: false
                    tasks:
                        - task: txtai.workflow.Task
                          action: nop
                          args: []
                          initialize: testworkflowfactory.initialize
                          finalize: testworkflowfactory.initialize
                index:
                    tasks:
                        - action: [index, upsert]
                          unpack: true
                transform:
                    tasks:
                        - action: transform
                          onetomany: true
                retrieve:
                    tasks:
                        - task: retrieve
                          safeopen: false
            """
        )
        expected = copy.deepcopy(config)

        for _ in range(2):
            app = Application(config)
            self.assertEqual(config, expected)
            self.assertEqual(list(app.workflow("template", ["x"])), ["HI X?"])
            self.assertEqual(list(app.workflow("stream", [3])), [0, 1, 2])
            self.assertIsInstance(app.workflows["stream"].stream, StreamTask)
            self.assertFalse(app.workflows["index"].tasks[0].unpack)
            self.assertEqual(app.workflows["index"].tasks[0].finalize, app.upsert)
            self.assertFalse(app.workflows["transform"].tasks[0].onetomany)
            self.assertFalse(app.workflows["retrieve"].tasks[0].safeinput.safeopen)

    def testWorkflowConfig(self):
        """
        Test repeated workflow construction preserves task and stream dictionaries
        """

        def action(values):
            return values

        stream = range
        config = {
            "tasks": [{"task": "txtai.workflow.Task", "action": action, "args": []}],
            "stream": {"task": "stream", "action": stream, "batch": False},
        }
        expected = copy.deepcopy(config)

        for _ in range(2):
            workflow = WorkflowFactory.create(config, "test")
            self.assertEqual(config, expected)
            self.assertEqual(list(workflow([3])), [0, 1, 2])
            self.assertIs(workflow.stream.action[0], stream)

    def testTaskConfig(self):
        """
        Test argument binding preserves caller configuration and object identities
        """

        marker = object()

        def action(values, argument):
            self.assertIs(argument, marker)
            return values

        for actions, args in [(action, [marker]), (action, {"argument": marker}), ([action], [[marker]]), ([action], [{"argument": marker}])]:
            config = {"action": actions, "args": args, "unpack": False}
            expected = config.copy()
            for _ in range(2):
                task = TaskFactory.create(config, "")
                self.assertEqual(config, expected)
                self.assertIs(config["action"], actions)
                self.assertIs(config["args"], args)
                self.assertEqual(task([1, 2]), [1, 2])


def initialize():
    """
    Test workflow lifecycle callback.
    """


def upper(values, suffix):
    """
    Test action with an additional argument.
    """

    return [value.upper() + suffix for value in values]
