"""
Defines API paths for upload endpoints.
"""

import shutil
import tempfile

from typing import List

from fastapi import APIRouter, File, Form, UploadFile

from ..route import EncodingAPIRoute


router = APIRouter(route_class=EncodingAPIRoute)


@router.post("/upload")
def upload(files: List[UploadFile] = File(), suffix: str = Form(default=None)):
    """
    Uploads files for local server processing.

    Args:
        data: list of files to upload

    Returns:
        list of server paths
    """

    paths = []
    for f in files:
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(f.file, tmp)
            paths.append(tmp.name)

    return paths
