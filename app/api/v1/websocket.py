"""WebSocket endpoint for real-time voice conversation."""

import asyncio
import base64
import contextlib
import json
import queue
import threading
from collections.abc import Iterator
from time import perf_counter

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from loguru import logger

from app.agent.graph_agent import _run_graph_and_get_state, stream_formatter_tokens
from app.core.config import get_settings
from app.db import queries
from app.services.voice import (
    TTS_REALTIME_FORMAT,
    MarkdownToSpeech,
    transcribe,
    tts_stream_sync,
)

router = APIRouter(tags=["websocket"])


async def _stream_realtime_tts(websocket: WebSocket, text_stream) -> str:
    """Stream LLM tokens to the client + pipe to ElevenLabs realtime TTS simultaneously.

    Returns the full assembled response text.

    Architecture (thread-queue bridge):

        LLM async token stream
              ├── websocket response_chunk  (text display)
              └── token_q.put(token)
                          │  threading.Queue bridges async ↔ sync
        ElevenLabs worker thread
          tts_stream_sync(token_iterator) → MP3 bytes
              └── audio_q.put_nowait(bytes)
                          │  asyncio.Queue
        drain_audio task
          → websocket audio_chunk

    No sentence buffering — ElevenLabs starts synthesising after the very first
    token (~150 ms to first audio chunk).
    """
    loop = asyncio.get_running_loop()
    token_q: queue.Queue[str | None] = queue.Queue()  # None = EOF sentinel
    audio_q: asyncio.Queue[bytes | None] = asyncio.Queue()  # None = EOF sentinel
    full_text = ""
    # Converts markdown tables → spoken numbered sentences; strips inline markers.
    # Text display uses raw tokens (markdown renders correctly in the UI).
    # ElevenLabs receives the spoken form so the voice reads naturally.
    md_converter = MarkdownToSpeech()

    # ── Sync token iterator consumed by ElevenLabs inside the worker thread ──
    def _token_iter() -> Iterator[str]:
        while True:
            tok = token_q.get()
            if tok is None:
                return
            yield tok

    # ── Background thread: feed tokens into ElevenLabs, push audio to async queue ──
    def _elevenlabs_thread() -> None:
        try:
            for chunk in tts_stream_sync(_token_iter()):
                if chunk:
                    loop.call_soon_threadsafe(audio_q.put_nowait, chunk)
        except Exception as e:
            logger.error(f"ElevenLabs realtime TTS error: {e}")
        finally:
            loop.call_soon_threadsafe(audio_q.put_nowait, None)  # EOF

    threading.Thread(target=_elevenlabs_thread, daemon=True).start()

    # ── Async task: drain audio queue → send audio_chunk messages ──
    async def _drain_audio() -> None:
        while True:
            chunk = await audio_q.get()
            if chunk is None:
                break
            await websocket.send_json(
                {
                    "type": "audio_chunk",
                    "data": base64.b64encode(chunk).decode("ascii"),
                }
            )

    drain_task = asyncio.create_task(_drain_audio())

    # ── Feed tokens: WebSocket display (raw) + ElevenLabs (spoken) ──
    try:
        async for chunk in text_stream:
            if chunk.startswith("__META__"):
                continue
            full_text += chunk
            # Send raw markdown to the UI — it renders correctly there.
            await websocket.send_json({"type": "response_chunk", "text": chunk})
            # Convert markdown → spoken text before sending to ElevenLabs.
            for spoken in md_converter.feed(chunk):
                token_q.put(spoken)
    finally:
        # Flush any buffered table or partial line at end of stream.
        for spoken in md_converter.flush():
            token_q.put(spoken)
        token_q.put(None)  # signal ElevenLabs the stream is done

    await drain_task  # wait for all audio to be sent before closing
    await websocket.send_json({"type": "response_end", "full_text": full_text})
    return full_text


@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    Real-time voice conversation over WebSocket.

    Protocol:
    1. Client connects with cookie auth (accessToken) and optional conversation_id query param
    2. Client sends audio chunks as binary frames
    3. After VAD fires, client sends {"type":"process"}
    4. Server transcribes → runs agent → streams text + TTS audio back

    Message Formats (Client → Server):
    - Binary:                    Raw audio data (accumulates until 'process')
    - {"type":"process"}         Process accumulated audio
    - {"type":"audio","data":"…"} Audio as base64
    - {"type":"ping"}            Keep-alive
    - {"type":"set_conversation","id":<n>}  Bind to existing conversation

    Message Formats (Server → Client):
    - {"type":"transcript","text":"…"}
    - {"type":"response_start"}
    - {"type":"response_chunk","text":"…"}
    - {"type":"response_end","full_text":"…"}
    - {"type":"audio_start"}
    - {"type":"audio_chunk","data":"<base64>","sentence_index":<n>}
    - {"type":"audio_end"}
    - {"type":"error","message":"…"}
    - {"type":"pong"}
    - {"type":"conversation_id","id":<n>}
    """
    settings = get_settings()
    # Cookies are not forwarded when the WebSocket connects to a different port
    # (browser treats localhost:3000 and localhost:8001 as separate origins).
    # Fall back to the token query param supplied by the /users/me/ws-token endpoint.
    token = websocket.cookies.get("accessToken") or websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Not authenticated")
        return

    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        external_id = payload["userId"]
        jwt_name = (
            payload.get("name")
            or payload.get("firstName")
            or payload.get("given_name")
            or ""
        )
        user = await queries.get_or_create_user(
            external_id, payload.get("email", ""), jwt_name
        )
        user_id = user["id"]
    except (ExpiredSignatureError, InvalidTokenError):
        await websocket.close(code=1008, reason="Invalid token")
        return

    await websocket.accept()

    audio_buffer = bytearray()

    conversation_id_param = websocket.query_params.get("conversation_id")
    if conversation_id_param:
        conversation_id = int(conversation_id_param)
    else:
        conv = await queries.create_conversation(user_id, "Voice Conversation")
        conversation_id = conv["id"]
        await websocket.send_json({"type": "conversation_id", "id": conversation_id})

    # Per-connection confirmation state — tracks multi-turn flows like
    # "Would you like me to create soccer?" → "Yes" across voice turns.
    voice_awaiting: str | None = None
    voice_context: dict | None = None

    # ── Main message loop ──────────────────────────────────────────────────────
    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                audio_buffer.extend(message["bytes"])
                continue

            if "text" in message:
                data = json.loads(message["text"])
                msg_type = data.get("type")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if msg_type == "audio":
                    audio_buffer.extend(base64.b64decode(data["data"]))
                    continue

                if msg_type == "set_conversation":
                    conversation_id = data.get("id")
                    voice_awaiting = None
                    voice_context = None
                    continue

                if msg_type == "process":
                    if not audio_buffer:
                        await websocket.send_json(
                            {"type": "error", "message": "No audio data to process"}
                        )
                        continue

                    # Client sends content_type so the server uses the correct
                    # file extension + MIME type for STT.  Silero VAD sends WAV;
                    # the old MediaRecorder path sent WebM.
                    content_type = data.get("content_type", "audio/webm")

                    try:
                        transcript = await transcribe(bytes(audio_buffer), content_type)
                        audio_buffer.clear()

                        if not transcript.strip():
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "message": "Could not transcribe audio",
                                }
                            )
                            continue

                        await websocket.send_json(
                            {"type": "transcript", "text": transcript}
                        )

                        await websocket.send_json({"type": "response_start"})
                        await websocket.send_json(
                            {"type": "audio_start", "format": TTS_REALTIME_FORMAT}
                        )

                        start = perf_counter()
                        state, response = await _run_graph_and_get_state(
                            transcript,
                            user_id,
                            conversation_id,
                            voice_awaiting,
                            voice_context,
                        )

                        # Capture awaiting state for the next voice turn so multi-turn
                        # confirmations ("create soccer?" → "Yes") work in voice mode.
                        if response.data and response.data.get("awaiting"):
                            voice_awaiting = response.data["awaiting"]
                            voice_context = dict(response.data)
                        else:
                            voice_awaiting = None
                            voice_context = None

                        async def _token_gen(
                            s=state,
                            r=response,
                            cid=conversation_id,
                            uid=user_id,
                            msg=transcript,
                            t=start,
                        ):
                            async for chunk in stream_formatter_tokens(
                                s, r, cid, uid, msg, t
                            ):
                                yield chunk

                        await _stream_realtime_tts(websocket, _token_gen())
                        await websocket.send_json({"type": "audio_end"})

                    except Exception as e:
                        logger.exception(
                            f"Voice processing error: {type(e).__name__}: {e}"
                        )
                        await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "message": str(e)})
