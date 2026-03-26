"""
SamiX Audio Intelligence Utility

This module provides high-level tools for audio manipulation and synthesis.
It is designed to handle the various audio formats encountered during 
customer support audits, ensuring they are standardized for transcription.

Key Capabilities:
1. Format Normalization: Converts MP3/FLAC/M4A/etc. to 16kHz Mono WAV.
2. Smart Summarization: Constructs human-readable scripts from audit data.
3. Text-to-Speech (TTS): Generates playable audio summaries for quick review.
"""
from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path
from typing import Optional


class AudioProcessor:
    """
    Orchestrates the audio lifecycle within the SamiX pipeline.
    
    It acts as a bridge between raw user uploads and the transcription/analysis 
    engines, ensuring audio quality and consistency.
    """

    # Target specifications for the Deepgram/Whisper transcription engines.
    TARGET_RATE:     int = 16_000
    TARGET_CHANNELS: int = 1

    # Audio Normalization (Standardization) 

    def convert_to_wav(
        self,
        file_bytes: bytes,
        filename: str,
    ) -> tuple[bytes, dict]:
        """
        Standardizes an audio file for downstream transcription.
        
        Uses pydub to convert any incoming format into a 16kHz Mono WAV file.
        Returns the raw WAV bytes and a metadata dictionary containing 
        the original file's characteristics.
        """
        meta = {
            "original_filename": filename,
            "original_format":   Path(filename).suffix.lstrip("."),
            "duration_sec": 0,
            "channels": 1,
            "frame_rate": self.TARGET_RATE,
            "pydub_used": False,
        }
        try:
            from pydub import AudioSegment

            # Detect format from extension or fallback to MP3.
            fmt = Path(filename).suffix.lstrip(".") or "mp3"
            seg = AudioSegment.from_file(io.BytesIO(file_bytes), format=fmt)

            meta["duration_sec"]   = round(len(seg) / 1000)
            meta["channels"]       = seg.channels
            meta["frame_rate"]     = seg.frame_rate
            meta["pydub_used"]     = True

            # Perform the conversion to target specs.
            seg = seg.set_frame_rate(self.TARGET_RATE).set_channels(self.TARGET_CHANNELS)

            buf = io.BytesIO()
            seg.export(buf, format="wav")
            return buf.getvalue(), meta
        except ImportError:
            # Fallback if ffmpeg/pydub is not in the environment.
            print("WARNING: pydub not installed - audio conversion skipped.")
            return file_bytes, meta
        except Exception as exc:
            print(f"WARNING: pydub conversion error ({exc}). Using original file.")
            return file_bytes, meta

    # Narrative Generation (Smart Summary) 

    def generate_text_summary(
        self,
        transcript_summary: str,
        key_moments: list[str],
        scores: Optional[dict] = None,
    ) -> str:
        """
        Constructs a conversational script for the 'Smart Summary' audio report.
        
        Synthesizes the transcript overview, identifies critical failures, 
        and reports the final audit status in a natural, spoken-word format.
        """
        lines = [
            "SamiX quality audit summary.",
            transcript_summary,
        ]
        if key_moments:
            lines.append("Key moments identified:")
            for moment in key_moments[:3]:
                lines.append(f"  {moment}")
        if scores:
            final = scores.get("final_score", 0)
            verdict = scores.get("verdict", "")
            lines.append(f"Final quality score: {final} out of 100. Verdict: {verdict}.")
            integrity = scores.get("integrity", 0)
            if integrity < 5:
                lines.append(
                    "Integrity dimension was critically low — "
                    "policy compliance issues were detected."
                )
        lines.append("End of summary.")
        return "  ".join(lines)

    def synthesise_audio(self, text: str) -> Optional[bytes]:
        """
        Generates a playable audio file from a text summary.
        
        Implements a tiered fallback strategy:
        1. Google TTS (gTTS): High-quality, requires internet access.
        2. pyttsx3: Standard quality, runs entirely offline.
        """
        result = self._gtts_synth(text)
        if result:
            return result
        return self._pyttsx3_synth(text)

    # TTS Implementation (Private) 

    @staticmethod
    def _gtts_synth(text: str) -> Optional[bytes]:
        """ Uses Google's online text-to-speech engine. """
        try:
            from gtts import gTTS
            # Limit to 3000 chars to avoid API timeouts.
            tts = gTTS(text=text[:3000], lang="en", slow=False)
            buf = io.BytesIO()
            tts.write_to_fp(buf)
            return buf.getvalue()
        except Exception:
            return None

    @staticmethod
    def _pyttsx3_synth(text: str) -> Optional[bytes]:
        """ Uses the local system's native TTS engine. """
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # pyttsx3 requires a physical file to save to before reading back.
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            engine.save_to_file(text[:3000], tmp_path)
            engine.runAndWait()
            with open(tmp_path, "rb") as fh:
                wav_bytes = fh.read()
            os.unlink(tmp_path)
            return wav_bytes
        except Exception:
            return None

    # Formatting Helpers 

    @staticmethod
    def duration_label(seconds: int) -> str:
        """ Formats a second count into a standard MM:SS timestamp. """
        m, s = divmod(int(seconds), 60)
        return f"{m}:{s:02d}"
