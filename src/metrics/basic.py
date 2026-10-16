"""기본 지표: 비음성 환각률(HR)과 WER.

HR 정의 (Calm-Whisper): 비음성 클립 중 출력 길이 > 0인 비율.
여기서는 앞뒤 공백을 지운 뒤 길이가 0보다 크면 환각으로 센다. 구두점만 있는 출력(예: ".")도 환각으로 센다.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List, Optional


def hallucination_rate(outputs: List[str]) -> Dict[str, float]:
    """비음성 클립 출력 목록 → 환각률과 개수."""
    n = len(outputs)
    hits = sum(1 for o in outputs if o.strip())
    return {"hr": hits / n if n else float("nan"), "n_hallucinated": hits, "n_total": n}


def basic_normalize(text: str) -> str:
    """소문자화, 구두점 제거, 공백 정리. Whisper 기준 수치와 비교할 때는 Whisper 정규화기를 쓴다."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return " ".join(text.split())


def wer(refs: List[str], hyps: List[str], normalize: Optional[Callable[[str], str]] = basic_normalize) -> Dict[str, float]:
    """코퍼스 단위 WER과 삽입·삭제·치환 개수. 삽입 개수는 IER 계산에 쓴다."""
    import jiwer

    if normalize:
        refs = [normalize(r) for r in refs]
        hyps = [normalize(h) for h in hyps]
    # 정답이 빈 문장이면 jiwer가 오류를 내므로 WER 계산에서는 제외한다 (비음성은 hallucination_rate로 따로 잰다)
    pairs = [(r, h) for r, h in zip(refs, hyps) if r]
    out = jiwer.process_words([r for r, _ in pairs], [h for _, h in pairs])
    n_ref = out.hits + out.substitutions + out.deletions
    return {
        "wer": out.wer,
        "ier": out.insertions / n_ref if n_ref else float("nan"),
        "insertions": out.insertions,
        "deletions": out.deletions,
        "substitutions": out.substitutions,
        "n_ref_words": n_ref,
        "n_utts": len(pairs),
    }
