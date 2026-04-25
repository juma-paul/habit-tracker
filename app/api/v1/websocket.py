"""WebSocket endpoint for real-time voice conversation."""

import base64
import json

import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI

from app.core.config import get_settings
from app.agent.agent import run_agent_stream
from app.services.voice import transcribe
from app.db import queries


router = APIRouter(tags=["websocket"])

async def get_tts_stream(client: AsyncOpenAI, text: str, settings):
    """Generate TTS audio and yield chunks."""
    response = await client.audio.speech.create(
        model=settings.tts_model,
        voice=settings.tts_voice,
        input=text,
        response_format="mp3"
    )
    return response.content

@router.websocket("/ws/voice")
async def voice_websocket(websocket: WebSocket):
    """
    Real-time voice conversation over WebSocket.

    Protocol:
    1. Client connects with token in query: /ws/voice?token=<jwt>&conversation_id=<id>
    2. Client sends audio chuncks as binary or base64 JSON
    3. Server streams back: transcription, agent response, and TTS audio

    Message Formats:

    Client -> Server
    - Binary: Raw audio data (accumulates until 'process' command)
    - JSON: {"type": "process"} - Process accumulated audio
    - JSON: {"type": "audio", "data": "<base64>"} - Audio as base64
    - JSON: {"type": "ping"} - Keep-alive
    - JSON: {"type": "set_conversation", "id": <number>} - Set conversation ID

    Server -> Client
    - {"type": "transcript", "text": "..."} - Transcript result
    - {"type": "response_start"} - Agent starting to respond
    - {"type": "response_chunk", "text": "..."} - Streaming text chunk
    - {"type": "response_end", "full_text": "..."} - Complete response
    - {"type": "audio_start"} - TTS audio starting
    - {"type": "audio_chunk", "data": "<base64>"} - TTS audio chunk
    - {"type": "audio_end"} - TTS audio complete
    - {"type": "error", "message": "..."} - Error occured
    - {"type": "pong"} - Keep-alive response
    - {"type": "conversation_id", "id": "<number>"} - Conversation ID for this session

    Example client (Javascript)
    ```javascript
    const ws = new WebSocket('ws://localhost:8001/api/v1/ws/voice?token=<jwt>');

    // Send audio
    mediaRecorder.ondataavailable = (e) => {
        ws.send(e.data) // Send binary audio
    };

    // Process when done recording
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        switch (msg.type) {
            case 'transcript':
                console.log('You said:', msg.text);
                break;
            case 'response_chunk':
                appendToUI(msg.text);
                break;
            case 'audio_chunk':
                playAudio(base64ToBlob(msg.data));
                break;
        }
    };
    ```
    """
    # verify accessToken cookie - browsers send cookies automatically on WS connect
    settings = get_settings()
    token = websocket.cookies.get("accessToken")
    if not token:
        await websocket.close(code=1008, reason="Not authenticated")
        return
    
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        external_id = payload["userId"]
        user = await queries.get_or_create_user(external_id, payload.get("email", ""))
        user_id = user["id"]
    except (ExpiredSignatureError, InvalidTokenError):
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    await websocket.accept()

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
    audio_buffer = bytearray()

    # Get or create conversation for voice session
    conversation_id_param = websocket.query_params.get("conversation_id")
    if conversation_id_param:
        conversation_id = int(conversation_id_param)
    else:
        # Create a new conversation for this voice session
        conv = await queries.create_conversation(user_id, "Voice Conversation")
        conversation_id = conv["id"]

        # Send conversation ID to client
        await websocket.send_json({"type": "conversation_id", "id": conversation_id})

        try:
            while True:
                # Receive message (binary or text)
                message = await websocket.receive()

                if "bytes" in message:
                    # Binary audio data - accumulate
                    audio_buffer.extend(message["bytes"])
                    continue

                if "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "ping":
                        await websocket.send_json({"type": "pong"})
                        continue

                    if msg_type == "audio":
                        # Base64 encoded audio
                        audio_bytes = base64.b64decode(data["data"])
                        audio_buffer.extend(audio_bytes)
                        continue

                    if msg_type == "set_conversation":
                        # Allow client to set conversation ID mid-session
                        conversation_id = data.get("id")
                        continue

                    if msg_type == "process":
                        if not audio_buffer:
                            await websocket.send_json({
                                "type": "error",
                                "message": "No audio data to process"
                            })
                            continue

                        try:
                            # Transcribe
                            transcript = await transcribe(bytes(audio_buffer), "audio/webm")
                            audio_buffer.clear()

                            if not transcript.strip():
                                await websocket.send_json({
                                    "type": "error",
                                    "message": "Could not transcribe audio"
                                })
                                continue

                            # Send transcript
                            await websocket.send_json({
                                "type": "transcript",
                                "text": transcript
                            })
                            
                            # Save user message to a conversation
                            await queries.add_message(conversation_id, "user", transcript)

                            # Stream agent response
                            await websocket.send_json({"type": "response_start"})

                            full_response = ""
                            async for chunk in run_agent_stream(transcript, user_id, conversation_id):
                                full_response += chunk
                                await websocket.send_json({
                                    "type": "response_chunk",
                                    "text": chunk
                                })

                            await websocket.send_json({
                                "type": "response_end",
                                "full_text": full_response
                            })

                            # Save assistant response to conversation
                            await queries.add_message(conversation_id, "assistant", full_response)

                            # Generate and send TTS audio
                            if full_response:
                                await websocket.send_json({"type": "audio_start"})

                                # Get TTS audio
                                audio_data = await get_tts_stream(client, full_response, settings)

                                # Send in chunks
                                chunk_size = 16 * 1024
                                for i in range(0, len(audio_data), chunk_size):
                                    chunk = audio_data[i:i + chunk_size]
                                    await websocket.send_json({
                                        "type": "audio_chunk",
                                        "data": base64.b64decode(chunk).decode()
                                    })

                                await websocket.send_json({"type": "audio_end"})

                        except Exception as e:
                            await websocket.send_json({
                                "type": "error",
                                "message": str(e)
                            })

        except WebSocketDisconnect:
            pass
        except Exception as e:
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
            except:
                pass