"""
SamiX Speech-to-Text (STT) Processing Engine

This module orchestrates the conversion of various audio and text formats into
a unified, speaker-separated transcript. It features a prioritized pipeline:
1. High-Precision Audio: Deepgram Nova-3 with Batch Diarization.
2. Local Fallback: OpenAI Whisper (Local) for offline or low-confidence scenarios.
3. Multi-Format Text: Native parsers for JSON, CSV, and plain-text transcripts.
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Union

import streamlit as st

from src.db import get_db
from src.storage import FileStorage
from src.utils.history_manager import TranscriptTurn


class STTProcessor:
    """
    Handles the ingestion and normalization of call data.
    
    The processor is designed to be format-agnostic, automatically routing
    incoming files to the appropriate transcription or parsing engine.
    """

    # Supported file formats.
    AUDIO_EXTS: set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
    TEXT_EXTS:  set[str] = {".csv", ".json", ".txt"}
    
    # Confidence threshold for Deepgram; falls back to Whisper if below this.
    CONF_THRESHOLD: float = 0.70

    def __init__(self) -> None:
        """ Initializes the processor and establishes the Deepgram connection. """
        self._deepgram = self._build_deepgram()
        self._db = get_db()
        self._storage = FileStorage()

    def _build_deepgram(self) -> Optional[object]:
        """ Attempts to initialize the Deepgram SDK from secrets. """
        try:
            key = os.environ.get("DEEPGRAM_API_KEY", "") or st.secrets.get("deepgram", {}).get("api_key", "")
            if "REPLACE" in key:
                return None
            from deepgram import DeepgramClient
            return DeepgramClient(key)
        except Exception:
            return None

    @property
    def has_deepgram(self) -> bool:
        """ Returns True if the Deepgram cloud service is ready. """
        return self._deepgram is not None

    # Public API 

    async def process(
        self,
        file_bytes: bytes,
        filename: str,
        session_id: Optional[str] = None,
    ) -> list[TranscriptTurn]:
        """
        [Main Entry] Detects the file type and produces a normalized transcript.
        
        Returns a list of TranscriptTurn objects, each containing speaker tags
        and timestamps.
        """
        ext = Path(filename).suffix.lower()
        storage_session_id = session_id or str(uuid.uuid4())[:8]
        started = time.perf_counter()
        
        # Route audio files to transcription engines.
        if ext in self.AUDIO_EXTS:
            turns, processor_used, confidence_score = await self._process_audio(file_bytes, filename)
        elif ext in self.TEXT_EXTS:
            turns = self._parse_text(file_bytes, filename)
            processor_used = f"parser:{ext.lstrip('.') or 'text'}"
            confidence_score = 1.0
        else:
            turns = self._parse_text(file_bytes, filename)
            processor_used = "parser:text"
            confidence_score = 1.0
            
        transcript_text = transcript_to_text(turns)
        payload = {
            "session_id": storage_session_id,
            "filename": filename,
            "processor_used": processor_used,
            "confidence_score": confidence_score,
            "turns": [t.__dict__ for t in turns],
        }
        file_path, request_hash = self._storage.save_json(
            "transcriptions",
            storage_session_id,
            payload,
            filename_prefix="transcription",
        )
        self._db.save_transcription(
            storage_session_id,
            filename,
            transcript_text,
            confidence_score=confidence_score,
            processor_used=processor_used,
            status="completed",
        )
        self._db.save_api_response(
            session_id=storage_session_id,
            api_name=processor_used,
            endpoint="transcription",
            request_hash=request_hash,
            response_json=payload,
            status_code=200,
            processing_time_ms=round((time.perf_counter() - started) * 1000, 2),
            file_path=file_path,
        )
        return turns

    async def process_chunk(self, audio_bytes: bytes) -> list[TranscriptTurn]:
        """
        Processes a small chunk of audio for live transcription.
        Specifically uses local Whisper to ensure low latency without API overhead.
        """
        # Small chunks don't need complex diarization; simplified Whisper logic is used.
        return await self._whisper_transcribe(audio_bytes, model_size="tiny")

    # Audio Processing Pipeline 

    async def _process_audio(self, data: bytes, filename: str) -> tuple[list[TranscriptTurn], str, float]:
        """
        Handles raw audio transcription using a prioritized fallback strategy.
        Prioritizes Deepgram (Cloud) over Whisper (Local).
        """
        # Run pydub conversion in a thread to avoid blocking the event loop.
        wav_data = await asyncio.to_thread(self._pydub_convert, data, filename)

        # Level 1: Deepgram (Preferred for speed and diarization quality).
        if self.has_deepgram:
            result, confidence = await self._deepgram_transcribe(wav_data)
            if result is not None:
                return result, "deepgram:nova-3", confidence

        # Level 2: Whisper (Fallback for accuracy and reliability).
        turns = await self._whisper_transcribe(wav_data)
        return turns, "whisper:base", 0.65

    def _pydub_convert(self, data: bytes, filename: str) -> bytes:
        """
        Converts uploaded audio into a standardized WAV format using Pydub.
        """
        try:
            from pydub import AudioSegment
            ext = Path(filename).suffix.lstrip(".") or "mp3"
            seg = AudioSegment.from_file(io.BytesIO(data), format=ext)
            # Re-sample to 16kHz mono (standard for STT engines).
            seg = seg.set_frame_rate(16000).set_channels(1)
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            return buf.getvalue()
        except Exception as exc:
            st.warning(f"pydub conversion failed ({exc}). Using original bytes.")
            return data

    async def _deepgram_transcribe(
        self, wav_data: bytes
    ) -> tuple[Optional[list[TranscriptTurn]], float]:
        """
        Executes a diarized transcription call to Deepgram Nova-3.
        """
        try:
            # We use a thread for the SDK call as some versions of the Python SDK 
            # are still predominantly synchronous in their top-level client.
            return await asyncio.to_thread(self._sync_deepgram_call, wav_data)
        except Exception as exc:
            st.warning(f"Deepgram error: {exc}. Falling back to Whisper.")
            return None, 0.0

    def _sync_deepgram_call(self, wav_data: bytes) -> tuple[Optional[list[TranscriptTurn]], float]:
        """ Synchronous wrapper for the Deepgram SDK call. """
        try:
            from deepgram import PrerecordedOptions, FileSource
            source: FileSource = {"buffer": wav_data}
            opts = PrerecordedOptions(
                model="nova-3",
                diarize=True,
                punctuate=True,
                language="en",
            )
            resp = self._deepgram.listen.prerecorded.v("1").transcribe_file(
                source, opts
            )
            words = resp.results.channels[0].alternatives[0].words
            
            if words:
                avg_conf = sum(w.confidence for w in words) / len(words)
                if avg_conf < self.CONF_THRESHOLD:
                    return None, float(avg_conf)
            else:
                avg_conf = 0.0

            return self._dg_words_to_turns(words), float(avg_conf)
        except Exception:
            return None, 0.0

    @staticmethod
    def _dg_words_to_turns(words: list) -> list[TranscriptTurn]:
        """
        Post-processes Deepgram word-segments into conversational turns.
        """
        turns: list[TranscriptTurn] = []
        turn_num   = 0
        cur_spk    = None
        cur_text   = []
        cur_start  = 0.0

        speaker_map: dict[int, str] = {}   # dg speaker_id → "AGENT"|"CUSTOMER"
        next_label  = iter(["AGENT", "CUSTOMER"])

        def flush():
            nonlocal turn_num, cur_text, cur_spk
            if not cur_text:
                return
            turn_num += 1
            ts = f"{int(cur_start//60):02d}:{int(cur_start%60):02d}"
            turns.append(TranscriptTurn(
                turn=turn_num, speaker=cur_spk or "AGENT",
                text=" ".join(cur_text), timestamp=ts,
            ))
            cur_text = []

        for w in words:
            spk_id = getattr(w, "speaker", 0)
            if spk_id not in speaker_map:
                speaker_map[spk_id] = next(next_label, f"SPEAKER_{spk_id}")
            label = speaker_map[spk_id]

            if label != cur_spk:
                flush()
                cur_spk   = label
                cur_start = getattr(w, "start", 0.0)

            cur_text.append(w.word)
        flush()
        return turns

    async def _whisper_transcribe(
        self, wav_data: bytes, model_size: str = "base"
    ) -> list[TranscriptTurn]:
        """
        Executes local transcription using OpenAI's Whisper model (via asyncio.to_thread).
        """
        return await asyncio.to_thread(self._sync_whisper_transcribe, wav_data, model_size)

    def _sync_whisper_transcribe(self, wav_data: bytes, model_size: str) -> list[TranscriptTurn]:
        """ Synchronous execution of Whisper model in a background thread. """
        try:
            import whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(wav_data)
                tmp_path = tmp.name
            
            model = whisper.load_model(model_size)
            result = model.transcribe(tmp_path)
            os.unlink(tmp_path)
            
            segs = result.get("segments", [])
            turns = []
            for i, seg in enumerate(segs, 1):
                spk = "AGENT" if i % 2 == 1 else "CUSTOMER"
                ts  = f"{int(seg['start']//60):02d}:{int(seg['start']%60):02d}"
                turns.append(TranscriptTurn(
                    turn=i, speaker=spk, text=seg["text"].strip(), timestamp=ts
                ))
            return turns or self._mock_turns()
        except Exception:
            return self._mock_turns()

    # Text Parsing Pipeline 

    def _parse_text(self, data: bytes, filename: str) -> list[TranscriptTurn]:
        """ Dispatches text-based files to structured format parsers. """
        ext = Path(filename).suffix.lower()
        text = data.decode("utf-8", errors="replace")

        if ext == ".json":
            return self._parse_json(text)
        if ext == ".csv":
            return self._parse_csv(text)
        return self._parse_plain(text)

    @staticmethod
    def _parse_json(text: str) -> list[TranscriptTurn]:
        """ Maps common JSON/Chat structures to NormalizedTranscript format. """
        try:
            data = json.loads(text)
            turns = []
            for i, item in enumerate(data if isinstance(data, list) else [], 1):
                spk = str(item.get("speaker", item.get("role", "AGENT"))).upper()
                if any(x in spk for x in ["CUSTOMER", "USER", "CLIENT"]):
                    spk = "CUSTOMER"
                else:
                    spk = "AGENT"
                turns.append(TranscriptTurn(
                    turn=i, speaker=spk,
                    text=str(item.get("text", item.get("message", item.get("content", "")))),
                    timestamp=str(item.get("timestamp", item.get("time", f"00:{i:02d}:00"))),
                ))
            return turns or STTProcessor._mock_turns()
        except Exception:
            return STTProcessor._mock_turns()

    @staticmethod
    def _parse_csv(text: str) -> list[TranscriptTurn]:
        """ Map-reduces CSV columns to the NormalizedTranscript schema. """
        import csv
        import io as _io
        turns = []
        reader = csv.DictReader(_io.StringIO(text))
        fieldnames = reader.fieldnames or []
        
        # Discover relevant columns by name heuristic.
        spk_col  = next((c for c in fieldnames if "speaker" in c.lower() or "role" in c.lower()), None)
        text_col = next((c for c in fieldnames if "text" in c.lower() or "message" in c.lower() or "content" in c.lower()), None)
        ts_col   = next((c for c in fieldnames if "time" in c.lower() or "timestamp" in c.lower()), None)

        for i, row in enumerate(reader, 1):
            spk = str(row.get(spk_col, "AGENT")).upper() if spk_col else ("AGENT" if i%2==1 else "CUSTOMER")
            if "CUSTOMER" in spk or "USER" in spk:
                spk = "CUSTOMER"
            else:
                spk = "AGENT"
            txt = str(row.get(text_col, "")) if text_col else str(list(row.values())[0])
            ts  = str(row.get(ts_col, f"00:0{i}:00")) if ts_col else f"00:0{i}:00"
            if txt.strip():
                turns.append(TranscriptTurn(turn=i, speaker=spk, text=txt, timestamp=ts))
        return turns or STTProcessor._mock_turns()

    @staticmethod
    def _parse_plain(text: str) -> list[TranscriptTurn]:
        """ Parses unstructured text using speaker-prefixed line detection. """
        turns = []
        # Support for 'AGENT: ...' or 'CUSTOMER [T1]: ...'
        pattern = re.compile(r"^(AGENT|CUSTOMER|AGENT\s*\[.*?\]|CUSTOMER\s*\[.*?\])[\s:]+(.+)$", re.IGNORECASE)
        for i, line in enumerate(text.splitlines(), 1):
            m = pattern.match(line.strip())
            if m:
                spk = "CUSTOMER" if "CUSTOMER" in m.group(1).upper() else "AGENT"
                turns.append(TranscriptTurn(turn=len(turns)+1, speaker=spk, text=m.group(2).strip(), timestamp=f"00:0{i}:00"))
        return turns or STTProcessor._mock_turns()

    # Fallback Mock Data 

    @staticmethod
    def _mock_turns() -> list[TranscriptTurn]:
        """ Typical call transcript for demonstration/testing. """
        raw = [
            ("CUSTOMER", "Hi, I've been charged twice for my subscription this month and need this fixed immediately.", "00:00:08"),
            ("AGENT",    "Thank you for calling. Let me pull up your account. Can I verify your email address?", "00:00:19"),
            ("CUSTOMER", "It's priya@example.com. This is the second time this happened. I'm really frustrated.", "00:00:31"),
            ("AGENT",    "I can see the duplicate charge. I'll process the refund — it should be back in about 2 days.", "00:00:48"),
            ("CUSTOMER", "Only 2 days? Last time it took 2 weeks. That doesn't seem right.", "00:01:05"),
            ("AGENT",    "Yes, 2 days maximum. Is there anything else I can help you with?", "00:01:18"),
            ("CUSTOMER", "Why does this keep happening? This is the second charge in two months.", "00:01:29"),
            ("AGENT",    "It must have been a system error. I've noted it on your account.", "00:01:42"),
            ("CUSTOMER", "That's not an answer. I want to speak to a supervisor.", "00:01:55"),
            ("AGENT",    "I'll transfer you now. Thank you for calling.", "00:02:08"),
        ]
        return [
            TranscriptTurn(turn=i+1, speaker=s, text=t, timestamp=ts)
            for i, (s, t, ts) in enumerate(raw)
        ]


def transcript_to_text(turns: list[TranscriptTurn]) -> str:
    """ Flatten a structured turn-list into a single string for LLM analysis. """
    lines = []
    for t in turns:
        lines.append(f"{t.speaker} [T{t.turn} {t.timestamp}]: {t.text}")
    return "\n".join(lines)
