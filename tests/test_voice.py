from unittest.mock import MagicMock, patch
import pytest
from src.document_ai.voice.transcriber import VoiceTranscriber

def _fake_segment(text, start=0.0, end=1.0):
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = text
    return seg

def _fake_info(language="en", probability=0.98, duration=2.5):
    info = MagicMock()
    info.language = language
    info.language_probability = probability
    info.duration = duration
    return info

@pytest.fixture(autouse=True)
def reset_model_cache():
    # The model is cached at class level; reset between tests so mocks don't leak.
    VoiceTranscriber._model = None
    yield
    VoiceTranscriber._model = None

def test_transcribe_joins_segments():
    with patch("document_ai.voice.transcriber.WhisperModel", create=True):
        transcriber = VoiceTranscriber.__new__(VoiceTranscriber)
        transcriber.model_size = "small"
        transcriber.device = "cpu"
        transcriber.compute_type = "int8"
        mock_model = MagicMock()
        segments = [_fake_segment(" Hello "), _fake_segment("world.")]
        mock_model.transcribe.return_value = (segments, _fake_info())
        VoiceTranscriber._model = mock_model
        result = transcriber.transcribe(b"fake-audio-bytes", filename="q.wav")
        assert result.text == "Hello world."
        assert result.language == "en"
        assert len(result.segments) == 2

def test_transcribe_rejects_empty_audio():
    transcriber = VoiceTranscriber.__new__(VoiceTranscriber)
    VoiceTranscriber._model = MagicMock()
    with pytest.raises(ValueError):
        transcriber.transcribe(b"", filename="q.wav")
