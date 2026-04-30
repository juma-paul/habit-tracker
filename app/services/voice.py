"""Voice services: transcription (whisper) and text-to-speech."""

import tempfile
from functools import lru_cache
from pathlib import Path

from openai import OpenAI

from app.core.config import get_settings


@lru_cache
def _get_client() -> OpenAI:
    """Lazy-load OpenAI client."""
    return OpenAI(api_key=get_settings().openai_api_key.get_secret_value())


async def transcribe(audio_data: bytes, content_type: str) -> str:
    """Transcribe audio to text using whisper."""
    extensions = {
        "audio/webm": ".webm",
        "audio/mp3": ".mp3",
        "audio/mpeg": ".mp3",
        "audio/wav": ".wav",
        "audio/m4a": ".m4a",
        "audio/mp4": ".m4a",
        "audio/ogg": ".ogg",
    }

    ext = extensions.get(content_type, ".webm")

    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_data)
        temp_path = Path(f.name)

    try:
        settings = get_settings()
        with open(temp_path, "rb") as audio_file:
            response = _get_client().audio.transcriptions.create(
                model=settings.whisper_model,
                file=audio_file,
                prompt="Habit tracking: walking, running, meditation. water, excercise, gym, steps, glasses, minutes",
            )
        return response.text
    finally:
        temp_path.unlink(missing_ok=True)


async def text_to_speech(text: str) -> bytes:
    """Convert text to speech using OpenAI TTS."""
    settings = get_settings()
    response = _get_client().audio.speech.create(
        model=settings.tts_model, voice=settings.tts_voice, input=text
    )
    return response.content
