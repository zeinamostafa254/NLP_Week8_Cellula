"""
voice/transcriber.py
---------------------
Speech-to-text for the "ask by voice" feature, backed by faster-whisper
(a CTranslate2 reimplementation of OpenAI Whisper — ~4x faster and much
lighter on memory than the original openai-whisper package, with no
extra system dependency beyond ffmpeg for audio decoding).
Usage:
    from document_ai.voice.transcriber import VoiceTranscriber
    transcriber = VoiceTranscriber()          # loads the model once
    result = transcriber.transcribe(audio_bytes, filename="query.wav")
    print(result.text)
The model is loaded lazily and cached as a module-level singleton so the
(relatively expensive) model load only happens once per process, the same
pattern used by the embedding model in ingestion/embedder.py.
"""
from __future__ import annotations
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from document_ai.config import (
    WHISPER_MODEL_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_LANGUAGE,
)
logger = logging.getLogger(__name__)

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str

@dataclass
class TranscriptionResult:
    text: str
    language: str
    language_probability: float
    duration: float
    segments: List[TranscriptSegment] = field(default_factory=list)

class VoiceTranscriber:
    """
    Thin wrapper around faster-whisper's WhisperModel.
    Model size / device / compute type are configurable via env vars
    (see config.py) so the same code runs fine on a laptop CPU
    (e.g. "small" + "int8") or a GPU box (e.g. "medium" + "float16").
    """
    _model = None  # class-level cache: shared across instances/requests
    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._ensure_model_loaded()

    def _ensure_model_loaded(self) -> None:
        if VoiceTranscriber._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is not installed. Run "
                "`uv add faster-whisper` (or `pip install faster-whisper`)."
            ) from e
        logger.info(
            f"Loading faster-whisper model '{self.model_size}' "
            f"(device={self.device}, compute_type={self.compute_type})…"
        )
        VoiceTranscriber._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        logger.info("faster-whisper model loaded.")

    def transcribe(
        self,
        audio: bytes,
        filename: str = "audio.wav",
        language: Optional[str] = WHISPER_LANGUAGE,
    ) -> TranscriptionResult:
        """
        Transcribe raw audio bytes (any format ffmpeg can decode: wav,
        mp3, m4a, webm/ogg from a browser mic recorder, etc.) into text.
        Args:
            audio: raw audio file bytes.
            filename: original filename, used only to infer a suffix so
                      ffmpeg can sniff the container format correctly.
            language: ISO-639-1 code to force a language (e.g. "en",
                      "ar"). Leave None to let Whisper auto-detect.
        """
        if not audio:
            raise ValueError("Received empty audio payload.")
        suffix = Path(filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
            tmp.write(audio)
            tmp.flush()
            segments_iter, info = VoiceTranscriber._model.transcribe(
                tmp.name,
                language=language,
                vad_filter=True,  # trims silence, helps with mic recordings
            )
            segments = [
                TranscriptSegment(start=s.start, end=s.end, text=s.text.strip())
                for s in segments_iter
            ]
        full_text = " ".join(s.text for s in segments).strip()
        logger.info(
            f"Transcribed {info.duration:.1f}s of audio "
            f"(detected language={info.language}, p={info.language_probability:.2f})."
        )
        return TranscriptionResult(
            text=full_text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=segments,
        )
