"""Colab에서 데이터를 받아 16kHz 모노 wav로 준비한다.

받은 원본과 변환본은 Drive(root)에 두어 세션이 끊겨도 다시 받지 않게 한다.
반환값은 (경로 목록, 정답 문장 목록). 비음성 클립의 정답은 빈 문자열이다.
"""
from __future__ import annotations

import io
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from .audio import prepare_clips

LIBRISPEECH_URL = "https://www.openslr.org/resources/12/{split}.tar.gz"
URBANSOUND_HF = "danavery/urbansound8K"  # 8,732클립, 원본과 같은 CC-BY-NC-4.0


def librispeech(root: str, split: str = "test-clean", n: Optional[int] = None) -> Tuple[List[str], List[str]]:
    """LibriSpeech 분할을 받아 앞에서부터 n개(None이면 전체) 발화를 준비한다."""
    root_p = Path(root)
    split_dir = root_p / "LibriSpeech" / split
    if not split_dir.exists():
        tar = root_p / f"{split}.tar.gz"
        root_p.mkdir(parents=True, exist_ok=True)
        if not tar.exists():
            subprocess.run(["wget", "-q", "-O", str(tar), LIBRISPEECH_URL.format(split=split)], check=True)
        subprocess.run(["tar", "-xzf", str(tar), "-C", str(root_p)], check=True)

    items = []
    for trans in sorted(split_dir.rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt_id, text = line.split(" ", 1)
            items.append((str(trans.parent / f"{utt_id}.flac"), text))
    if n is not None:
        items = items[:n]

    # LibriSpeech는 이미 16kHz 모노지만, 모든 모델에 같은 형식(wav)을 주기 위해 똑같이 변환한다
    paths = prepare_clips([p for p, _ in items], str(root_p / "prepared" / split))
    return paths, [t for _, t in items]


def urbansound8k(root: str, per_class: Optional[int] = None, seed: int = 0) -> Tuple[List[str], List[str], List[str]]:
    """UrbanSound8K를 받아 클래스별 per_class개(None이면 전체)를 준비한다.

    반환: (경로 목록, 정답(전부 빈 문자열), 클래스 이름 목록)
    """
    import random

    import soundfile as sf
    from datasets import Audio, load_dataset

    raw_dir = Path(root) / "urbansound8k_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # decode=False: datasets의 오디오 디코더(버전마다 의존성이 다름)를 쓰지 않고 바이트를 직접 읽는다
    ds = load_dataset(URBANSOUND_HF, split="train").cast_column("audio", Audio(decode=False))

    by_class = defaultdict(list)
    for i, c in enumerate(ds["class"]):
        by_class[c].append(i)
    rng = random.Random(seed)
    chosen = []
    for c in sorted(by_class):
        idx = by_class[c]
        chosen += idx if per_class is None else rng.sample(idx, min(per_class, len(idx)))

    src_paths, classes = [], []
    for i in chosen:
        ex = ds[i]
        dst = raw_dir / ex["slice_file_name"]
        if not dst.exists():
            audio, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]))
            sf.write(dst, audio, sr)
        src_paths.append(str(dst))
        classes.append(ex["class"])

    paths = prepare_clips(src_paths, str(Path(root) / "prepared" / "urbansound8k"))
    return paths, [""] * len(paths), classes
