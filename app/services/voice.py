"""Voice services — STT provider routing + ElevenLabs TTS.

STT_PROVIDER controls which engine transcribes audio:
  whisper     OpenAI Whisper API  — reliable, ~$0.006/min,  ~800ms latency
  groq        Groq Whisper        — same model, ~$0.00067/min, ~150ms latency
  elevenlabs  ElevenLabs Scribe   — ~$0.0067/min, ~400ms latency

TTS is ElevenLabs eleven_flash_v2_5.
- text_to_speech()   — whole-sentence call, returns bytes (kept for fallback use)
- tts_stream_sync()  — realtime streaming via convert_realtime; call from a
                       background thread; yields MP3 bytes as ElevenLabs
                       synthesises them (~150ms to first audio chunk).
"""

import asyncio
import re
import tempfile
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from openai import OpenAI

from app.core.config import get_settings

# Audio format used for realtime streaming.
# PCM = no MP3 encoder delay (~26 ms silence per chunk that causes audible gaps).
# pcm_24000 = 24 kHz, 16-bit, mono — good quality/bandwidth tradeoff.
# Must match the format string sent in the "audio_start" WebSocket message so
# the client knows how to decode the raw bytes without calling decodeAudioData.
TTS_REALTIME_FORMAT = "pcm_24000"
TTS_REALTIME_SAMPLE_RATE = 24_000

# Optimised voice settings for conversational real-time streaming:
# - stability 0.5  → natural variation without instability between short chunks
# - similarity 0.75 → faithful to voice without extra compute overhead
# - style 0        → must be 0 for streaming; any value adds latency
# - speaker_boost off → also adds latency; not needed for conversational use
_REALTIME_VOICE_SETTINGS = VoiceSettings(
    stability=0.5,
    similarity_boost=0.75,
    style=0.0,
    use_speaker_boost=False,
)

# ── Extension map shared by all providers ─────────────────────────────────────

_EXTENSIONS: dict[str, str] = {
    "audio/webm": ".webm",
    "audio/mp3": ".mp3",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/m4a": ".m4a",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
}


def _write_temp(audio_data: bytes, content_type: str) -> Path:
    ext = _EXTENSIONS.get(content_type, ".webm")
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
        f.write(audio_data)
        return Path(f.name)


# ── Provider clients (lazy, cached) ───────────────────────────────────────────


@lru_cache
def _get_openai() -> OpenAI:
    settings = get_settings()
    assert settings.openai_api_key, "OPENAI_API_KEY required for STT_PROVIDER=whisper"
    return OpenAI(api_key=settings.openai_api_key.get_secret_value())


@lru_cache
def _get_groq() -> OpenAI:
    """Groq uses the OpenAI-compatible API — no extra SDK needed."""
    settings = get_settings()
    assert settings.groq_api_key, "GROQ_API_KEY required for STT_PROVIDER=groq"
    return OpenAI(
        api_key=settings.groq_api_key.get_secret_value(),
        base_url="https://api.groq.com/openai/v1",
    )


@lru_cache
def _get_elevenlabs() -> ElevenLabs:
    return ElevenLabs(api_key=get_settings().elevenlabs_api_key.get_secret_value())


# ── STT implementations ────────────────────────────────────────────────────────


def _transcribe_whisper(temp_path: Path, content_type: str) -> str:
    settings = get_settings()
    # Pass file as (filename, bytes, mime_type) tuple — OpenAI SDK v2.x needs
    # the MIME type explicit or it may reject valid webm with "invalid file format".
    with open(temp_path, "rb") as f:
        response = _get_openai().audio.transcriptions.create(
            model=settings.whisper_model,
            file=(temp_path.name, f, content_type),
            language="en",
            prompt="Habit tracking: walking, running, meditation, water, exercise, gym, steps",
        )
    return response.text


def _transcribe_groq(temp_path: Path, content_type: str) -> str:
    with open(temp_path, "rb") as f:
        response = _get_groq().audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=(temp_path.name, f, content_type),
            language="en",
        )
    return response.text


def _transcribe_elevenlabs(temp_path: Path, content_type: str) -> str:  # noqa: ARG001
    with open(temp_path, "rb") as f:
        response = _get_elevenlabs().speech_to_text.convert(
            file=f,
            model_id="scribe_v1",
            language_code="en",
            tag_audio_events=False,
        )
    return response.text


# ── Public API ────────────────────────────────────────────────────────────────


async def transcribe(audio_data: bytes, content_type: str) -> str:
    """Transcribe audio using the configured STT_PROVIDER."""
    settings = get_settings()
    temp_path = _write_temp(audio_data, content_type)

    _providers = {
        "whisper": _transcribe_whisper,
        "groq": _transcribe_groq,
        "elevenlabs": _transcribe_elevenlabs,
    }
    fn = _providers.get(settings.stt_provider, _transcribe_whisper)

    try:
        return await asyncio.to_thread(fn, temp_path, content_type)
    finally:
        temp_path.unlink(missing_ok=True)


async def text_to_speech(text: str) -> bytes:
    """Convert text to speech using ElevenLabs eleven_flash_v2_5."""
    settings = get_settings()

    def _synth() -> bytes:
        chunks = _get_elevenlabs().text_to_speech.convert(
            text=text,
            voice_id=settings.elevenlabs_voice_id,
            model_id="eleven_flash_v2_5",
            output_format="mp3_44100_128",
        )
        return b"".join(chunks)

    return await asyncio.to_thread(_synth)


def _normalize_cell(value: str) -> str:
    """Convert a raw table cell value to spoken form."""
    if value.strip() in ("—", "-", "–", "", "N/A", "n/a", "null", "None"):
        return "not set"
    return value.strip()


def _table_lines_to_speech(lines: list[str]) -> str:
    """Convert a list of markdown table lines to a numbered spoken description.

    Input:
        ["| Habit | Target | Frequency | Streak |",
         "| --- | --- | --- | --- |",
         "| walking | No target set | Daily | — |",
         "| water drinking | 12 glasses | Daily | — |"]

    Output:
        "1. Habit: walking. Target: No target set. Frequency: Daily. Streak: not set.
         2. Habit: water drinking. Target: 12 glasses. Frequency: Daily. Streak: not set."
    """

    def parse_cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip("|").split("|")]

    def is_separator(cells: list[str]) -> bool:
        return all(re.fullmatch(r"[-: ]+", c) for c in cells if c)

    headers: list[str] = []
    data_rows: list[list[str]] = []

    for line in lines:
        cells = parse_cells(line)
        if is_separator(cells):
            continue
        if not headers:
            headers = cells
        else:
            data_rows.append(cells)

    if not headers or not data_rows:
        return ""

    parts: list[str] = []
    for i, row in enumerate(data_rows, 1):
        fields = []
        for header, value in zip(headers, row, strict=False):
            spoken = _normalize_cell(value)
            fields.append(f"{header}: {spoken}")
        parts.append(f"{i}. {'. '.join(fields)}.")

    return " ".join(parts)


def _strip_inline_markdown(text: str) -> str:
    """Remove inline markdown symbols from a line of prose."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)   # **bold**
    text = re.sub(r"\*(.*?)\*", r"\1", text)         # *italic*
    text = re.sub(r"`([^`]+)`", r"\1", text)         # `code`
    text = re.sub(r"^\s*[-*]\s+", "", text)          # bullet - or *
    text = re.sub(r"^\s*#{1,6}\s+", "", text)        # ### headings
    return text


class MarkdownToSpeech:
    """Line-buffered markdown → speech converter for the realtime TTS token stream.

    Prose tokens are passed through immediately (inline markdown stripped).
    Table lines are buffered until the table ends, then emitted as numbered
    spoken sentences so the voice reads "1. Habit: walking. Target: ..." instead
    of reading raw pipe characters and dashes.
    """

    def __init__(self) -> None:
        self._line_buf = ""
        self._table_lines: list[str] = []
        self._in_table = False

    def feed(self, chunk: str) -> Iterator[str]:
        """Yield spoken text as complete lines are received."""
        self._line_buf += chunk
        while "\n" in self._line_buf:
            line, self._line_buf = self._line_buf.split("\n", 1)
            yield from self._process_line(line)

    def flush(self) -> Iterator[str]:
        """Flush any buffered content at end of stream."""
        if self._line_buf:
            yield from self._process_line(self._line_buf)
            self._line_buf = ""
        if self._in_table and self._table_lines:
            spoken = _table_lines_to_speech(self._table_lines)
            if spoken:
                yield spoken
            self._table_lines = []
            self._in_table = False

    def _process_line(self, line: str) -> Iterator[str]:
        stripped = line.strip()
        if stripped.startswith("|"):
            self._in_table = True
            self._table_lines.append(stripped)
        else:
            if self._in_table:
                spoken = _table_lines_to_speech(self._table_lines)
                if spoken:
                    yield spoken + " "
                self._table_lines = []
                self._in_table = False
            spoken = _strip_inline_markdown(line)
            if spoken.strip():
                yield spoken + " "


def _context_buffered(token_iter: Iterator[str], min_chars: int = 30) -> Iterator[str]:
    """Buffer initial tokens until we have enough phonetic context.

    ElevenLabs commits to the first phoneme with whatever text it has when
    synthesis starts.  Feeding a single 1-3 char token causes misclassification
    at word boundaries (e.g. 'reading' → 'dreading') because 'r' and 'dr' are
    acoustically close and the model has no prior context.  Accumulating ~30
    chars before the first yield gives the model a full phrase to work with.
    """
    buf = ""
    for tok in token_iter:
        buf += tok
        if len(buf) >= min_chars:
            yield buf
            buf = ""
            break
    if buf:
        yield buf
    yield from token_iter


def tts_stream_sync(token_iter: Iterator[str]) -> Iterator[bytes]:
    """Realtime TTS via ElevenLabs convert_realtime — call from a background thread.

    Accepts a sync token iterator (e.g. fed from a threading.Queue) and yields
    raw PCM bytes as ElevenLabs synthesises them.  First audio arrives ~150ms
    after the first ~30 chars are buffered (see _context_buffered).
    """
    settings = get_settings()
    yield from _get_elevenlabs().text_to_speech.convert_realtime(
        voice_id=settings.elevenlabs_voice_id,
        text=_context_buffered(token_iter),
        model_id="eleven_flash_v2_5",
        output_format=TTS_REALTIME_FORMAT,
        voice_settings=_REALTIME_VOICE_SETTINGS,
    )
