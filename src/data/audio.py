"""오디오 전처리: 모든 모델에 같은 입력을 주기 위한 공통 함수.

Whisper-LLaDA의 inference.py는 리샘플·모노 변환을 하지 않으므로 (docs/setup_drax_whisper_llada.md 3절)
모든 클립을 미리 16kHz 모노 wav로 바꿔 두고, 모든 모델에 같은 파일을 넣는다.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

TARGET_SR = 16000


def load_mono_16k(path: str) -> np.ndarray:
    """오디오를 읽어 float32 모노, 16kHz로 돌려준다."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)  # 스테레오 → 모노 (채널 평균, Drax 방식과 동일)
    if sr != TARGET_SR:
        from scipy.signal import resample_poly

        g = np.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)
    return audio


def pad_to_min(audio: np.ndarray, min_sec: float = 1.0, sr: int = TARGET_SR) -> np.ndarray:
    """min_sec보다 짧으면 뒤를 0으로 채운다 (Whisper-LLaDA 학습 조건과 동일)."""
    n = int(min_sec * sr)
    if len(audio) >= n:
        return audio
    return np.concatenate([audio, np.zeros(n - len(audio), dtype=audio.dtype)])


def prepare_clips(src_paths: Iterable[str], out_dir: str, min_sec: Optional[float] = 1.0) -> List[str]:
    """클립들을 16kHz 모노 wav로 변환해 out_dir에 저장하고 새 경로 목록을 돌려준다.

    이미 변환된 파일은 건너뛰므로 Colab 세션이 끊겨도 이어서 실행할 수 있다.
    """
    import soundfile as sf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for p in src_paths:
        dst = out / (Path(p).stem + ".wav")
        if not dst.exists():
            audio = load_mono_16k(p)
            if min_sec:
                audio = pad_to_min(audio, min_sec)
            sf.write(dst, audio, TARGET_SR, subtype="PCM_16")
        new_paths.append(str(dst))
    return new_paths


def write_llada_annotation(paths: List[str], texts: List[str], out_json: str) -> None:
    """Whisper-LLaDA inference.py 입력 형식 {"annotation": [{"path", "text"}]}으로 저장한다."""
    ann = [{"path": p, "text": t} for p, t in zip(paths, texts)]
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"annotation": ann}, f, ensure_ascii=False, indent=1)
