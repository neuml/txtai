"""
Defines an OpenAI-compatible API endpoint for txtai.

See the following specification for more information:
https://github.com/openai/openai-openapi
"""

import uuid
import json
import time

from typing import List, Optional, Union

from fastapi import APIRouter, Body, Form, UploadFile
from fastapi.responses import Response, StreamingResponse

from .. import application
from ..route import EncodingAPIRoute

router = APIRouter(route_class=EncodingAPIRoute)


# pylint: disable=W0622
@router.post("/v1/chat/completions")
def chat(
    messages: List[dict] = Body(...),
    model: str = Body(...),
    max_completion_tokens: Optional[int] = Body(default=None),
    stream: Optional[bool] = Body(default=False),
):
    """
    Runs a chat completion request.

    Args:
        messages: list of messages [{"role": role, "content": content}]
        model: agent name, workflow name, pipeline name or embeddings
        max_completion_tokens: sets the max length to generate
        stream: streams response if True

    Returns:
        chat completion
    """

    # Build keyword arguments
    kwargs = {key: value for key, value in [("stream", stream), ("maxlength", max_completion_tokens)] if value}

    # Get first message
    message = messages[0]["content"]

    # Agent
    if model in application.get().agents:
        result = application.get().agent(model, message, **kwargs)

    # Embeddings search
    elif model == "embeddings":
        result = application.get().search(message, 1, **kwargs)[0]["text"]

    # Pipeline
    elif model in application.get().pipelines and model != "llm":
        result = application.get().pipeline(model, message, **kwargs)

    # Workflow
    elif model in application.get().workflows:
        result = list(application.get().workflow(model, [message], **kwargs))[0]

    # Default to running all messages through default LLM
    else:
        result = application.get().pipeline("llm", messages, **kwargs)

    # Write response
    return StreamingResponse(StreamingChatResponse()(model, result)) if stream else ChatResponse()(model, result)


@router.post("/v1/embeddings")
def embeddings(input: Union[str, List[str]] = Body(...), model: str = Body(...)):
    """
    Creates an embeddings vector for the input text.

    Args:
        input: text|list
        model: model name

    Returns:
        list of embeddings vectors
    """

    # Convert to embeddings
    result = application.get().batchtransform([input] if isinstance(input, str) else input)

    # Build and return response
    data = []
    for index, embedding in enumerate(result):
        data.append({"object": "embedding", "embedding": embedding, "index": index})

    return {"object": "list", "data": data, "model": model}


@router.post("/v1/audio/speech")
def speech(input: str = Body(...), voice: str = Body(...), response_format: Optional[str] = Body(default="mp3")):
    """
    Generates speech for the input text.

    Args:
        input: input text
        voice: speaker name
        response_format: audio encoding format, defaults to mp3

    Returns:
        audio data
    """

    # Convert to audio
    audio = application.get().pipeline("texttospeech", input, speaker=voice, encoding=response_format)

    # Write audio
    return Response(audio)


@router.post("/v1/audio/transcriptions")
def transcribe(file: UploadFile, language: Optional[str] = Form(default=None), response_format: Optional[str] = Form(default="json")):
    """
    Transcribes audio to text.

    Args:
        file: audio input file
        language: language of input audio
        response_format: output format (json or text)

    Returns:
        transcribed text
    """

    # Transcribe
    text = application.get().pipeline("transcription", file.file, language=language, task="transcribe")
    return text if response_format == "text" else {"text": text}


@router.post("/v1/audio/translations")
def translate(
    file: UploadFile,
    response_format: Optional[str] = Form(default="json"),
):
    """
    Translates audio to English.

    Args:
        file: audio input file
        response_format: output format (json or text)

    Returns:
        translated text
    """

    # Transcribe and translate to English
    text = application.get().pipeline("transcription", file.file, language="English", task="translate")
    return text if response_format == "text" else {"text": text}


class ChatResponse:
    """
    Returns a chat response object.
    """

    def __call__(self, model, result):
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time() * 1000),
            "model": model,
            "choices": [{"id": 0, "message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
        }


class StreamingChatResponse:
    """
    Returns a streaming chat response object.
    """

    def __call__(self, model, result):
        for chunk in result:
            yield "data: " + json.dumps(
                {
                    "id": str(uuid.uuid4()),
                    "object": "chat.completion.chunk",
                    "created": int(time.time() * 1000),
                    "model": model,
                    "choices": [{"id": 0, "delta": {"content": chunk}}],
                }
            ) + "\n\n"

        yield "data: [DONE]\n\n"
