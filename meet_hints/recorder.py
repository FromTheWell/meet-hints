import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np
import pyaudiowpatch as pyaudio


class AudioRecorder:
    """
    Captures system audio via WASAPI loopback + optionally a microphone,
    then mixes both streams into a single mono 16kHz WAV.
    """

    CHUNK_SIZE = 512
    FORMAT = pyaudio.paInt16

    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 16000,
        device_index: int | None = None,
        mic_device_index: int | None = None,
    ):
        self.output_dir = Path(output_dir).resolve() / "audio"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.mic_device_index = mic_device_index

        self._recording = False
        self._loopback_frames: list[bytes] = []
        self._mic_frames: list[bytes] = []
        self._loopback_info: dict = {}
        self._mic_info: dict = {}
        self._loopback_stream = None
        self._mic_stream = None
        self._loopback_thread: threading.Thread | None = None
        self._p: pyaudio.PyAudio | None = None
        self._output_path: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_loopback_devices(self) -> list[dict]:
        p = pyaudio.PyAudio()
        try:
            return [
                p.get_device_info_by_index(i)
                for i in range(p.get_device_count())
                if p.get_device_info_by_index(i).get("isLoopbackDevice", False)
            ]
        finally:
            p.terminate()

    def list_input_devices(self) -> list[dict]:
        p = pyaudio.PyAudio()
        try:
            return [
                p.get_device_info_by_index(i)
                for i in range(p.get_device_count())
                if p.get_device_info_by_index(i).get("maxInputChannels", 0) > 0
                and not p.get_device_info_by_index(i).get("isLoopbackDevice", False)
            ]
        finally:
            p.terminate()

    def start(self) -> Path:
        """Open streams and start capturing. Returns the output path immediately."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_path = self.output_dir / f"meeting_{timestamp}.wav"
        self._loopback_frames = []
        self._mic_frames = []

        self._p = pyaudio.PyAudio()

        loopback_info = self._resolve_loopback(self._p)
        mic_info = self._resolve_mic(self._p) if self.mic_device_index is not None else None

        self._loopback_info = loopback_info
        self._mic_info = mic_info or {}

        print(f"[DEBUG] Loopback: {loopback_info['name']} — "
              f"{int(loopback_info['maxInputChannels'])}ch "
              f"{int(loopback_info['defaultSampleRate'])}Hz "
              f"idx={int(loopback_info['index'])}")
        if mic_info:
            print(f"[DEBUG] Mic: {mic_info['name']} — "
                  f"{int(mic_info['maxInputChannels'])}ch "
                  f"{int(mic_info['defaultSampleRate'])}Hz "
                  f"idx={int(mic_info['index'])}")

        self._recording = True

        # Use blocking reads in a thread — WASAPI loopback in callback mode
        # does not fire callbacks during silence, so blocking reads are more reliable.
        self._loopback_stream = self._p.open(
            format=self.FORMAT,
            channels=int(loopback_info["maxInputChannels"]),
            rate=int(loopback_info["defaultSampleRate"]),
            input=True,
            input_device_index=int(loopback_info["index"]),
            frames_per_buffer=self.CHUNK_SIZE,
        )
        print("[DEBUG] Loopback stream open.")

        def _loopback_worker():
            while self._recording:
                try:
                    data = self._loopback_stream.read(
                        self.CHUNK_SIZE, exception_on_overflow=False
                    )
                    self._loopback_frames.append(data)
                except Exception:
                    break

        self._loopback_thread = threading.Thread(target=_loopback_worker, daemon=True)
        self._loopback_thread.start()

        if mic_info:
            def _mic_cb(in_data, frame_count, time_info, status):
                if self._recording:
                    self._mic_frames.append(in_data)
                    return (None, pyaudio.paContinue)
                return (None, pyaudio.paComplete)

            self._mic_stream = self._p.open(
                format=self.FORMAT,
                channels=min(int(mic_info["maxInputChannels"]), 2),
                rate=int(mic_info["defaultSampleRate"]),
                input=True,
                input_device_index=int(mic_info["index"]),
                frames_per_buffer=self.CHUNK_SIZE,
                stream_callback=_mic_cb,
            )
            self._mic_stream.start_stream()
            print("[DEBUG] Mic stream open.")

        return self._output_path

    def stop(self) -> Path:
        self._recording = False

        if self._loopback_thread is not None:
            self._loopback_thread.join(timeout=3)
            self._loopback_thread = None

        for stream in (self._loopback_stream, self._mic_stream):
            if stream is not None:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass

        self._loopback_stream = None
        self._mic_stream = None

        if self._p is not None:
            try:
                self._p.terminate()
            except Exception:
                pass
            self._p = None

        print(f"[DEBUG] Captured {len(self._loopback_frames)} loopback frames, "
              f"{len(self._mic_frames)} mic frames.")

        if not self._loopback_frames and not self._mic_frames:
            raise RuntimeError("No audio frames captured from any device.")

        if not self._loopback_frames:
            print("[WARNING] No loopback frames captured — saving mic-only audio. "
                  "Loopback only records audio that plays through the speakers; "
                  "make sure meeting audio was active during recording.")

        self._save_wav()
        return self._output_path

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    def _resolve_loopback(self, p: pyaudio.PyAudio) -> dict:
        loopbacks = [
            p.get_device_info_by_index(i)
            for i in range(p.get_device_count())
            if p.get_device_info_by_index(i).get("isLoopbackDevice", False)
        ]

        print(f"[DEBUG] Found {len(loopbacks)} loopback devices: "
              f"{[int(d['index']) for d in loopbacks]}")

        if self.device_index is not None:
            match = next((d for d in loopbacks if int(d["index"]) == self.device_index), None)
            if match:
                return match
            raise RuntimeError(
                f"Loopback device index {self.device_index} not found. "
                f"Available: {[int(d['index']) for d in loopbacks]}"
            )

        try:
            return p.get_default_wasapi_loopback_device()
        except Exception:
            pass

        if loopbacks:
            return loopbacks[0]

        raise RuntimeError("No WASAPI loopback device found.")

    def _resolve_mic(self, p: pyaudio.PyAudio) -> dict:
        return p.get_device_info_by_index(self.mic_device_index)

    # ------------------------------------------------------------------
    # Audio processing
    # ------------------------------------------------------------------

    def _to_mono_16k(self, frames: list[bytes], device_info: dict) -> np.ndarray:
        channels = (
            int(device_info.get("maxInputChannels") or 0)
            or int(device_info.get("maxOutputChannels") or 0)
            or 2
        )
        rate = int(device_info.get("defaultSampleRate", 48000))

        audio = np.frombuffer(b"".join(frames), dtype=np.int16)

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

        if rate != self.sample_rate:
            ratio = rate / self.sample_rate
            new_len = int(len(audio) / ratio)
            indices = np.round(np.linspace(0, len(audio) - 1, new_len)).astype(int)
            audio = audio[indices]

        return audio

    def _save_wav(self):
        if self._loopback_frames:
            loopback_audio = self._to_mono_16k(self._loopback_frames, self._loopback_info)
        else:
            loopback_audio = None

        if self._mic_frames and self._mic_info:
            mic_audio = self._to_mono_16k(self._mic_frames, self._mic_info)
        else:
            mic_audio = None

        if loopback_audio is not None and mic_audio is not None:
            min_len = min(len(loopback_audio), len(mic_audio))
            mixed = np.clip(
                loopback_audio[:min_len].astype(np.int32) + mic_audio[:min_len].astype(np.int32),
                -32768, 32767,
            ).astype(np.int16)
        elif loopback_audio is not None:
            mixed = loopback_audio
        else:
            mixed = mic_audio

        with wave.open(str(self._output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(mixed.tobytes())
