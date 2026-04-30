"""Voice endpoints: speech-to-text and text-to-speech."""

from urllib.parse import quote

from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.agent.agent import run_agent
from app.services.voice import transcribe, text_to_speech
from app.models.schemas import VoiceResponse
from app.api.deps import CurrentUser


router = APIRouter(prefix="/voice", tags=["voice"])
limiter = Limiter(key_func=get_remote_address)

SUPPORTED_TYPES = {
    "audio/webm",
    "audio/mp3",
    "audio/mpeg",
    "audio/wav",
    "audio/m4a",
    "audio/mp4",
    "audio/ogg",
}
MAX_SIZE = 25 * 1024 * 1024


@router.post("/chat", response_model=VoiceResponse)
@limiter.limit("10/minute")
async def voice_chat(
    request: Request,
    user_id: CurrentUser,
    audio: UploadFile = File(...),
) -> VoiceResponse:
    """
    Process voice input and return text + audio response.

    1. Transcribe audio to text (Whisper)
    2. Process with AI agent
    3. Return transcript, agent response, and TTS audio URL
    """
    # Validate file type
    base_type = (audio.content_type or "").split(";")[0].strip()
    if base_type not in SUPPORTED_TYPES:
        raise HTTPException(
            400, f"Unsupported format. Use: {', '.join(SUPPORTED_TYPES)}"
        )

    # Read and validate size
    data = await audio.read()
    if len(data) > MAX_SIZE:
        raise HTTPException(400, f"File too large. Max: {MAX_SIZE // (1024 * 1024)}MB")

    if len(data) == 0:
        raise HTTPException(400, "Empty audio file")

    # Transcribe
    transcript = await transcribe(data, audio.content_type or "")

    # Process with agent
    agent_response = await run_agent(transcript, user_id)

    return VoiceResponse(
        transcript=transcript,
        agent_response=agent_response,
        audio_url=f"/api/v1/voice/tts?text={quote(agent_response.message[:200])}",
    )


@router.get("/tts")
async def get_tts(text: str) -> Response:
    """Convert text to speech and return audio."""
    if len(text) > 1000:
        raise HTTPException(400, "Text too long. Max 1000 characters.")

    audio_data = await text_to_speech(text)

    return Response(
        content=audio_data,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline; filename=response.mp3"},
    )


@router.post("/transcribe")
async def transcribe_only(audio: UploadFile = File(...)) -> dict:
    """Transcribe audio to text only (for testing)."""
    base_type = (audio.content_type or "").split(";")[0].strip()
    if base_type not in SUPPORTED_TYPES:
        raise HTTPException(400, "Unsupported format")

    data = await audio.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty audio file")

    text = await transcribe(data, audio.content_type or "")
    return {"transcript": text}
