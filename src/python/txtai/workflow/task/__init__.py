"""
Task imports
"""

from .base import Task
from .console import ConsoleTask
from .export import ExportTask
from .factory import TaskFactory
from .file import FileTask
from .image import ImageTask
from .retrieve import RetrieveTask
from .service import ServiceTask
from .storage import StorageTask
from .stream import StreamTask
from .template import *
from .template import RagTask as ExtractorTask
from .url import UrlTask
from .workflow import WorkflowTask
