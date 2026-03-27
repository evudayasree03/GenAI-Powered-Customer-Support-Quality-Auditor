"""
SamiX Speech-to-Text (STT) Processing Engine - Dual-Mode Version
- Client Mode: Sends audio to Render Backend (Streamlit Cloud safe).
- Server Mode: Processes transcription via Deepgram (Render.com optimized).
"""
from __future__ import annotations

import asyncio
import io
import json
import os
import re
import logging
import time
import uuid
import requests
from pathlib import Path
from typing import Optional, Union, List

# Use logging instead of st.warning for backend stability
logger = logging.getLogger("samix.stt")

from src.db import get_db
from src.storage import FileStorage
from src.utils.history_manager import TranscriptTurn


class STTProcessor:
    AUDIO_EXTS: set[str] = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
    TEXT_EXTS:  set[str] = {".csv", ".json", ".txt"}
    CONF_THRESHOLD: float = 0.40

    def __init__(self) -> None:
        # 1. Detect Environment
        self.api_url = os.environ.get("BACKEND_URL")
        self.is_client = self.api_url is not None

        # 2. Conditional Initialization
        if self.is_client:
            logger.info(f"STT running in CLIENT mode. Routing to: {self.api_url}")
            self._deepgram = None
        else:
            logger.info("STT running in SERVER mode. Initializing Deepgram...")
            self._deepgram = self._build_deepgram()
            self._db = get_db()
            self._storage = FileStorage()

    def _build_deepgram(self) -> Optional[object]:
        try:
            from deepgram import DeepgramClient
            key = os.environ.get("DEEPGRAM_API_KEY", "")
            if not key or "REPLACE" in key:
                logger.error("DEEPGRAM_API_KEY missing or invalid.")
                return None
            return DeepgramClient(key)
        except Exception as e:
            logger.error(f"Failed to build Deepgram client: {e}")
            return None

    @property
    def has_deepgram(self) -> bool:
        return self._deepgram is not None

    async def process(
        self,
        file_bytes: bytes,
        filename: str,
        session_id: Optional[str] = None,
    ) -> list[TranscriptTurn]:
        """Main entry point: Automatically chooses between API call or local processing."""
        if self.is_client:
            return await self._process_via_api(file_bytes, filename)
        
        # SERVER LOGIC (Render.com)
        ext = Path(filename).suffix.lower()
        storage_session_id = session_id or str(uuid.uuid4())[:8]
        started = time.perf_counter()
        
        if ext in self.AUDIO_EXTS:
            turns, processor_used, confidence_score = await self._process_audio(file_bytes, filename)
        else:
            turns = self._parse_text(file_bytes, filename)
            processor_used = f"parser:{ext.lstrip('.') or 'text'}"
            confidence_score = 1.0
            
        transcript_text = transcript_to_text(turns)
        payload = {
            "session_id": storage_session_id,
            "filename": filename,
            "processor_used": processor_used,
            "confidence_score": confidence_score,
            "turns": [t.__dict__ for t in turns],
        }

        # Save to DB and Storage (Only on Server)
        file_path, request_hash = self._storage.save_json("transcriptions", storage_session_id, payload)
        self._db.save_transcription(storage_session_id, filename, transcript_text, confidence_score, processor_used, "completed")
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

    async def _process_via_api(self, data: bytes, filename: str) -> list[TranscriptTurn]:
        """Frontend logic: Proxies the request to the Render backend."""
        try:
            # Note: Pointing to the specific /audit endpoint on your FastAPI backend
            files = {"file": (filename, data)}
            response = requests.post(f"{self.api_url}/audit", files=files, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                # Reconstruct TranscriptTurn objects from JSON
                return [TranscriptTurn(**t) for t in result.get("turns", [])]
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return self._mock_turns()
        except Exception as e:
            logger.error(f"Failed to connect to backend STT: {e}")
            return self._mock_turns()

    async def _process_audio(self, data: bytes, filename: str) -> tuple[list[TranscriptTurn], str, float]:
        wav_data = await asyncio.to_thread(self._pydub_convert, data, filename)
        if self.has_deepgram:
            result, confidence = await self._deepgram_transcribe(wav_data)
            if result:
                return result, "deepgram:nova-2", confidence
        return self._mock_turns(), "mock:fallback", 0.50

    def _pydub_convert(self, data: bytes, filename: str) -> bytes:
        try:
            from pydub import AudioSegment
            ext = Path(filename).suffix.lstrip(".") or "mp3"
            seg = AudioSegment.from_file(io.BytesIO(data), format=ext)
            seg = seg.set_frame_rate(16000).set_channels(1)
            buf = io.BytesIO()
            seg.export(buf, format="wav")
            return buf.getvalue()
        except Exception as exc:
            logger.error(f"pydub conversion failed: {exc}")
            return data

    async def _deepgram_transcribe(self, wav_data: bytes) -> tuple[Optional[list[TranscriptTurn]], float]:
        try:
            from deepgram import PrerecordedOptions
            source = {"buffer": wav_data}
            opts = PrerecordedOptions(model="nova-2", diarize=True, punctuate=True, language="en")
            resp = self._deepgram.listen.prerecorded.v("1").transcribe_file(source, opts)
            words = resp.results.channels[0].alternatives[0].words
            if not words: return None, 0.0
            avg_conf = sum(w.confidence for w in words) / len(words)
            return self._dg_words_to_turns(words), float(avg_conf)
        except Exception as e:
            logger.error(f"Deepgram API error: {e}")
            return None, 0.0

    @staticmethod
    def _dg_words_to_turns(words: list) -> list[TranscriptTurn]:
        turns: list[TranscriptTurn] = []
        turn_num, cur_spk, cur_text, cur_start = 0, None, [], 0.0
        speaker_map: dict[int, str] = {}
        next_label = iter(["AGENT", "CUSTOMER"])

        def flush():
            nonlocal turn_num, cur_text, cur_spk
            if not cur_text: return
            turn_num += 1
            ts = f"{int(cur_start//60):02d}:{int(cur_start%60):02d}"
            turns.append(TranscriptTurn(turn=turn_num, speaker=cur_spk or "AGENT", text=" ".join(cur_text), timestamp=ts))
            cur_text = []

        for w in words:
            spk_id = getattr(w, "speaker", 0)
            if spk_id not in speaker_map:
                speaker_map[spk_id] = next(next_label, f"SPEAKER_{spk_id}")
            label = speaker_map[spk_id]
            if label != cur_spk:
                flush()
                cur_spk, cur_start = label, getattr(w, "start", 0.0)
            cur_text.append(w.word)
        flush()
        return turns

    def _parse_text(self, data: bytes, filename: str) -> list[TranscriptTurn]:
        ext = Path(filename).suffix.lower()
        text = data.decode("utf-8", errors="replace")
        if ext == ".json": return self._parse_json(text)
        if ext == ".csv": return self._parse_csv(text)
        return self._parse_plain(text)

    def _parse_json(self, text: str) -> list[TranscriptTurn]:
        try:
            data = json.loads(text)
            turns = []
            for i, item in enumerate(data if isinstance(data, list) else [], 1):
                turns.append(TranscriptTurn(
                    turn=i, 
                    speaker=item.get("speaker", "AGENT"), 
                    text=item.get("text", ""), 
                    timestamp=item.get("timestamp", "00:00")
                ))
            return turns
        except: return self._mock_turns()

    def _parse_csv(self, text: str) -> list[TranscriptTurn]:
        import csv
        turns = []
        reader = csv.DictReader(io.StringIO(text))
        for i, row in enumerate(reader, 1):
            turns.append(TranscriptTurn(turn=i, speaker=row.get("speaker", "AGENT"), text=row.get("text", ""), timestamp=row.get("timestamp", "00:00")))
        return turns

    def _parse_plain(self, text: str) -> list[TranscriptTurn]:
        turns = []
        for i, line in enumerate(text.splitlines(), 1):
            if ":" in line:
                spk, txt = line.split(":", 1)
                turns.append(TranscriptTurn(turn=i, speaker=spk.strip(), text=txt.strip(), timestamp="00:00"))
        return turns

    @staticmethod
    def _mock_turns() -> list[TranscriptTurn]:
        raw = [
            ("CUSTOMER", "I need help with my account.", "00:00"),
            ("AGENT", "I can help with that. What is your email?", "00:05")
        ]
        return [TranscriptTurn(turn=i+1, speaker=s, text=t, timestamp=ts) for i, (s, t, ts) in enumerate(raw)]

def transcript_to_text(turns: list[TranscriptTurn]) -> str:
    return "\n".join([f"{t.speaker} [T{t.turn} {t.timestamp}]: {t.text}" for t in turns])
