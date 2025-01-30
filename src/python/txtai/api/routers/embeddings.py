"""
Defines API paths for embeddings endpoints.
"""

from io import BytesIO
from typing import List, Optional

import PIL

from fastapi import APIRouter, Body, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder

from .. import application
from ..responses import ResponseFactory
from ..route import EncodingAPIRoute

from ...app import ReadOnlyError
from ...graph import Graph

router = APIRouter(route_class=EncodingAPIRoute)


@router.get("/search")
def search(query: str, request: Request):
    """
    Finds documents most similar to the input query. This method will run either an index search
    or an index + database search depending on if a database is available.

    Args:
        query: input query
        request: FastAPI request

    Returns:
        list of {id: value, score: value} for index search, list of dict for an index + database search
    """

    # Execute search
    results = application.get().search(query, request=request)

    # Encode using standard FastAPI encoder but skip certain classes
    results = jsonable_encoder(
        results, custom_encoder={bytes: lambda x: x, BytesIO: lambda x: x, PIL.Image.Image: lambda x: x, Graph: lambda x: x.savedict()}
    )

    # Return raw response to prevent duplicate encoding
    response = ResponseFactory.create(request)
    return response(results)


# pylint: disable=W0621
@router.post("/batchsearch")
def batchsearch(
    request: Request,
    queries: List[str] = Body(...),
    limit: int = Body(default=None),
    weights: float = Body(default=None),
    index: str = Body(default=None),
    parameters: List[dict] = Body(default=None),
    graph: bool = Body(default=False),
):
    """
    Finds documents most similar to the input queries. This method will run either an index search
    or an index + database search depending on if a database is available.

    Args:
        queries: input queries
        limit: maximum results
        weights: hybrid score weights, if applicable
        index: index name, if applicable
        parameters: list of dicts of named parameters to bind to placeholders
        graph: return graph results if True

    Returns:
        list of {id: value, score: value} per query for index search, list of dict per query for an index + database search
    """

    # Execute search
    results = application.get().batchsearch(queries, limit, weights, index, parameters, graph)

    # Encode using standard FastAPI encoder but skip certain classes
    results = jsonable_encoder(
        results, custom_encoder={bytes: lambda x: x, BytesIO: lambda x: x, PIL.Image.Image: lambda x: x, Graph: lambda x: x.savedict()}
    )

    # Return raw response to prevent duplicate encoding
    response = ResponseFactory.create(request)
    return response(results)


@router.post("/add")
def add(documents: List[dict] = Body(...)):
    """
    Adds a batch of documents for indexing.

    Args:
        documents: list of {id: value, text: value, tags: value}
    """

    try:
        application.get().add(documents)
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.post("/addobject")
def addobject(data: List[bytes] = File(), uid: List[str] = Form(default=None), field: str = Form(default=None)):
    """
    Adds a batch of binary documents for indexing.

    Args:
        data: list of binary objects
        uid: list of corresponding ids
        field: optional object field name
    """

    if uid and len(data) != len(uid):
        raise HTTPException(status_code=422, detail="Length of data and document lists must match")

    try:
        # Add objects
        application.get().addobject(data, uid, field)
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.post("/addimage")
def addimage(data: List[UploadFile] = File(), uid: List[str] = Form(), field: str = Form(default=None)):
    """
    Adds a batch of images for indexing.

    Args:
        data: list of images
        uid: list of corresponding ids
        field: optional object field name
    """

    if uid and len(data) != len(uid):
        raise HTTPException(status_code=422, detail="Length of data and uid lists must match")

    try:
        # Add images
        application.get().addobject([PIL.Image.open(content.file) for content in data], uid, field)
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.get("/index")
def index():
    """
    Builds an embeddings index for previously batched documents.
    """

    try:
        application.get().index()
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.get("/upsert")
def upsert():
    """
    Runs an embeddings upsert operation for previously batched documents.
    """

    try:
        application.get().upsert()
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.post("/delete")
def delete(ids: List = Body(...)):
    """
    Deletes from an embeddings index. Returns list of ids deleted.

    Args:
        ids: list of ids to delete

    Returns:
        ids deleted
    """

    try:
        return application.get().delete(ids)
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.post("/reindex")
def reindex(config: dict = Body(...), function: str = Body(default=None)):
    """
    Recreates this embeddings index using config. This method only works if document content storage is enabled.

    Args:
        config: new config
        function: optional function to prepare content for indexing
    """

    try:
        application.get().reindex(config, function)
    except ReadOnlyError as e:
        raise HTTPException(status_code=403, detail=e.args[0]) from e


@router.get("/count")
def count():
    """
    Total number of elements in this embeddings index.

    Returns:
        number of elements in embeddings index
    """

    return application.get().count()


@router.post("/explain")
def explain(query: str = Body(...), texts: List[str] = Body(default=None), limit: int = Body(default=None)):
    """
    Explains the importance of each input token in text for a query.

    Args:
        query: query text
        texts: list of text

    Returns:
        list of dict where a higher scores represents higher importance relative to the query
    """

    return application.get().explain(query, texts, limit)


@router.post("/batchexplain")
def batchexplain(queries: List[str] = Body(...), texts: List[str] = Body(default=None), limit: int = Body(default=None)):
    """
    Explains the importance of each input token in text for a query.

    Args:
        query: query text
        texts: list of text

    Returns:
        list of dict where a higher scores represents higher importance relative to the query
    """

    return application.get().batchexplain(queries, texts, limit)


@router.get("/transform")
def transform(text: str, category: Optional[str] = None, index: Optional[str] = None):
    """
    Transforms text into an embeddings array.

    Args:
        text: input text
        category: category for instruction-based embeddings
        index: index name, if applicable

    Returns:
        embeddings array
    """

    return application.get().transform(text, category, index)


@router.post("/batchtransform")
def batchtransform(texts: List[str] = Body(...), category: Optional[str] = None, index: Optional[str] = None):
    """
    Transforms list of text into embeddings arrays.

    Args:
        texts: list of text
        category: category for instruction-based embeddings
        index: index name, if applicable

    Returns:
        embeddings arrays
    """

    return application.get().batchtransform(texts, category, index)
