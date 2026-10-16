"""모델별 추론 래퍼의 공통 인터페이스.

모든 래퍼는 같은 입력(오디오 경로 목록)을 받아 같은 형식(TranscriptionResult 목록)을 돌려준다.
디코딩 설정은 결과마다 함께 저장한다 (CLAUDE.md 0절 규칙).
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DecodeConfig:
    """결과 파일에 함께 기록할 디코딩 설정. 해당 없는 값은 None으로 둔다."""

    model: str
    language: Optional[str] = "en"
    temperature: Optional[float] = 0.0
    beam_size: Optional[int] = None
    num_steps: Optional[int] = None      # 확산/flow 단계 수 N
    num_candidates: Optional[int] = None  # 후보 수 K
    selection: Optional[str] = None       # 후보 선택 방식 (mbr, confidence 등)
    seed: Optional[int] = 0
    extra: Dict[str, Any] = field(default_factory=dict)  # 모델 고유 설정 (block_length 등)


@dataclass
class TranscriptionResult:
    audio_path: str
    transcript: str
    candidates: List[str] = field(default_factory=list)            # RQ3: 후보 목록
    intermediate_steps: List[Dict[str, Any]] = field(default_factory=list)  # RQ2: 단계별 중간 출력
    elapsed_sec: Optional[float] = None
    config: Optional[DecodeConfig] = None


class BaseRunner(ABC):
    def __init__(self, config: DecodeConfig):
        self.config = config

    @abstractmethod
    def load(self) -> None:
        """모델과 가중치를 메모리에 올린다."""

    @abstractmethod
    def transcribe(self, audio_paths: List[str]) -> List[TranscriptionResult]:
        """오디오 경로 목록을 전사한다."""


def save_results(results: List[TranscriptionResult], path: str) -> None:
    """결과를 JSONL로 저장한다. 한 줄에 클립 하나, 설정 포함."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
