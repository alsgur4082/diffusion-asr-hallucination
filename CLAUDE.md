# 인수인계: 확산(diffusion) 디코더 ASR의 환각 연구

> 작성일: 2026-10-16 · 음성인식 과목 학부 연구 (약 2개월, Colab Pro+)
> Claude Code에서 이어서 진행하기 위한 문서다. 프로젝트 루트에 `CLAUDE.md`로 두면 자동으로 읽힌다.
> 연구계획서(RP) 초안: https://claude.ai/code/artifact/7159af76-8b49-48ce-8d37-d30b4cbd159a

---

## 0. 작업 규칙 (사용자 선호)

- 답변과 설명은 **한국어**로 한다. 처음 보는 사람도 따라올 수 있게 풀어서 설명한다.
- 사용자는 **Git 초급자**다. Git 명령을 쓸 때는 무엇을 하는 명령인지 한 줄씩 설명한다.
- 작업 방식은 **뼈대 먼저, 세부는 나중에**(scaffold first)다.
- 출처 없는 숫자를 만들지 않는다. 추정치는 "추정"이라고 표시하고, 실측하면 이 문서의 해당 값을 갱신한다.
- 모든 실험은 디코딩 설정(언어, temperature, beam, 단계 수 N, 후보 수 K, 시드)을 결과 파일에 함께 기록한다.

---

## 1. 연구 한 줄 요약

같은 Whisper-large-v3 인코더를 쓰는 자기회귀(AR)·확산·flow matching 디코더를 비교하고, 한 모델 안에서 디코딩 방식을 바꿔가며, **디코딩 방식이 ASR 환각에 미치는 영향**을 측정한다.

- 배경: 확산 ASR(Whisfusion, Drax, Whisper-LLaDA, MDM-ASR)은 정확도·속도만 보고했고, 환각은 아무도 측정하지 않았다(2026-10 문헌 검색 기준).
- 핵심 설계: Whisper-LLaDA는 설정만으로 **순수 확산 ↔ semi-AR** 디코딩을 바꿀 수 있다. 같은 모델·같은 학습 데이터에서 디코딩 방식만 바꾼 비교가 가능하다.

### 연구 질문
- **RQ1:** 디코딩 방식이 환각률과 환각 유형(반복 루프, 짧은 끼워넣기, 길이 폭주)을 바꾸는가?
- **RQ2:** 확산 디코더 안에서 환각 토큰은 몇 번째 디노이징 단계에서 생기고, 이후 지워지는가? 단계 수 N, 후보 수 K, 후보 선택 방식(MBR·신뢰도)의 영향은?
- **RQ3:** 확산 디코더의 후보 간 불일치를 정답 없는 환각 탐지 점수로 쓸 수 있는가? (Whisper no-speech 확률, 평균 log-prob, 압축률과 AUROC 비교)

---

## 2. 비교 설계

| 비교 | 모델 | 바뀌는 것 | 고정되는 것 |
|---|---|---|---|
| ① 모델 내 다이얼 (**핵심**) | Whisper-LLaDA: 순수 확산(`block_length == gen_len`) ↔ semi-AR(블록 64, 32) | 디코딩 방식 | 모델, 가중치, 학습 데이터 |
| ② 같은 인코더, 다른 디코더 | Whisper-large-v3(AR) vs Drax(flow matching) vs Whisper-LLaDA(확산) | 디코더 구조 + 학습 데이터 | 인코더 |
| ③ 보조 | Whisper-small(AR) vs Whisfusion v1(확산) | 디코더 구조 + 학습 데이터 | 인코더(Whisper-small) |

비교 ②·③은 학습 데이터 차이가 섞여 있으므로 결론은 ① 중심으로 낸다.

---

## 3. 모델 정보 (2026-10-16 확인)

### 3.1 Whisper-large-v3 (AR 기준)
- HF: `openai/whisper-large-v3` (transformers)
- 비교 ③용: `openai/whisper-small`

### 3.2 Drax (flow matching)
- HF: https://huggingface.co/aiola/drax-v1 · GitHub: https://github.com/aiola-lab/drax
- 구조: Whisper-large-v3 인코더(**동결**, 논문) + DiT 디코더, 약 1.2B, 라이선스 모델 카드 CC-BY-4.0 / **코드 CC-BY-NC-4.0**
- 언어: en, it, pt, de, fr, es, ja, zh (**한국어 없음**)
- 사용 예 (모델 카드 기준):
  ```python
  from drax import Transcriber
  asr = Transcriber(model_path="aiola/drax-v1")
  result = asr.transcribe("/path/to/audio.wav", language="en", sampling_steps=32, temperature=1e-2)
  print(result[0].transcript)
  ```
- 설치: `pip install -e .` (transformers==4.52.3 고정, torch 2.7.0 권장). 배치 추론 지원. 세부는 `docs/setup_drax_whisper_llada.md`
- 학습 데이터(논문): 8개 언어 약 15K시간. LibriSpeech, MLS, AMI, **Earnings-22**, VoxPopuli, TED-LIUM, CommonVoice-13 등 → Earnings-22 조건은 Drax에게 in-domain일 수 있음
- 중간 단계: `asr.solver.sample(..., return_intermediates=True, time_grid=...)`로 꺼낼 수 있음(코드 기준, 실행 미확인). 소스 분포가 uniform이라 토큰이 단계마다 바뀔 수 있음. 샘플링이 항상 확률적이므로 시드 기록 필수
- **미확인:** 메모리·속도 실측, 중간 단계 추출 코드 실제 동작.

### 3.3 Whisper-LLaDA (마스크 확산 LLM, 핵심 모델)
- GitHub: https://github.com/liuzhan22/Diffusion-ASR (ICASSP 2026, Apache-2.0)
- 구조: Whisper-large-v3 인코더 + LLaDA-8B-Instruct, 약 8.7B
- 설치 (README 기준):
  ```bash
  conda env create -f environment.yaml
  conda activate whisper-llada
  pip install k2==1.24.4.dev20250714+cuda12.6.torch2.7.1 -f https://k2-fsa.github.io/k2/installation/pre-compiled-cuda-wheels-linux/2.7.1.html
  ```
- 필요 가중치: `openai/whisper-large-v3`, `GSAI-ML/LLaDA-8B-Instruct`, 그리고 학습된 체크포인트(Google Drive):
  https://drive.google.com/file/d/1btAsCXTHQApnRcl_k5Wfn0zSD08yXHZh/view
- 추론: `python3 inference.py --cfg-path configs/decode_config.yaml` (**배치 추론 미지원**)
- 디코딩 모드 (`configs/decode_config.yaml`의 `decode` 섹션):
  - 순수 확산: `mode: "decoding"`, `block_length == gen_len` (예: 128)
  - semi-AR: `mode: "decoding"`, `block_length`를 `gen_len`의 정수 분의 1로 (예: 64, 32)
- 참고: Whisfusion v2 논문의 측정으로는 H100에서 RTFx 7.92로 비교 모델 중 가장 느렸다.
- 확인됨 (코드·논문): Whisper 인코더 **동결**, LLaDA는 동결 + LoRA(q/k/v, rank 8), Q-Former 어댑터 학습. 학습 데이터 **LibriSpeech 960h**(속도 변형 0.9/1.1). LLaDA는 fp16, Whisper 인코더는 fp32로 로드.
- Colab: conda 대신 pip(torch 2.7.1, transformers 4.53.2, **peft 0.3.0**). 추론에는 k2 불필요. `decode_config.yaml` 기본 모드가 `semi_ar_deliberation`이므로 `decoding`으로 바꿔야 함. `inference.py`는 리샘플·모노 변환을 안 하므로 16kHz 모노로 미리 변환. 세부는 `docs/setup_drax_whisper_llada.md`
- ⚠ **연구 설계 영향:** `decoding` 모드는 한 번 확정한 토큰을 다시 마스킹하지 않음 → RQ2 "이후 지워지는가"는 구조상 항상 아니오. temperature 0·low_confidence 고정 → RQ3 후보 K개를 만들려면 코드 수정 필요. 출력 상한 128토큰.
- **미확인:** 실제 메모리 사용량(가중치만 약 19GB로 추정), 클립당 시간, 체크포인트 로드 시 누락 키 여부, 체크포인트 파일 크기.

### 3.4 Whisfusion v1 (보조)
- GitHub: https://github.com/taeyoun811/Whisfusion · HF: `taeyoun811/whisfusion` (Apache-2.0)
- 구조: Whisper-small 인코더(88.2M) + SMDM-170M 확산 디코더 + 교차 어텐션 어댑터(42.5M). **LibriSpeech 960h만으로 학습, 영어 전용.**
- 가중치 파일: `whisfusion_stage2_decoder.pt`(전체), `whisfusion_stage1_adapter.pt`
- 설치 조건: torch 2.4.1(cu121) 고정, FlashAttention v2.6.3 소스 빌드(rotary, layer_norm, xentropy 포함). **T4 불가**, A100 또는 L4 사용.
- 평가: `scripts/05_evaluate_whisfusion.sh`, `scripts/06_evaluate_whisper.sh`
- 설치 실패 시 비교 ③은 제외해도 연구는 성립한다.

### 3.5 Whisfusion v2 (사용 불가)
- 논문(arXiv 2508.07048v2, 2026-06)은 Whisper-large-v3 인코더, 11개 언어, 한국어 지원판의 가중치 공개를 언급했지만, **GitHub·HF에는 v1만 있다.**
- 저자 연락처: ty8352@snu.ac.kr. 가중치를 받으면 한국어 축과 3자 비교를 추가한다.

---

## 4. 데이터셋

| 데이터 | 용도 | 비고 |
|---|---|---|
| UrbanSound8K | 비음성 환각률 | 8,732클립, 최대 4초. Calm-Whisper 기준 재현용 |
| ESC-50 | 비음성 환각률 교차 확인 | 2,000클립 |
| MUSAN (noise, music) | 저SNR 잡음 혼합, 비음성 | speech 부분은 쓰지 않음 |
| LibriSpeech test-clean / test-other | 정상 음성 정확도, 무음·잡음 합성 원본 | |
| Earnings-22 | 실제 회의 음성의 삽입 오류 | HF `distil-whisper/earnings22` |

- 비음성 입력은 정답이 빈 문장이므로 라벨링이 필요 없다.
- Whisper-LLaDA는 느리므로 비음성 데이터는 클래스별 계층 추출로 약 1,000클립까지 줄여 쓴다(1주차 실측 후 확정).
- 데이터셋 라이선스는 실제로 받을 때 확인한다.

---

## 5. 평가 지표와 기준 수치

- **비음성 환각률(HR):** 출력 길이 > 0인 클립 비율 (Calm-Whisper 정의)
- **IER(삽입 오류율), WER:** 무음, 잡음, 실제 음성 조건
- **환각 유형 비율:** 반복 루프, 상위 빈출 문구(BoH식 목록), 길이 폭주
- **탐지:** AUROC, 정밀도, 재현율

**기준 수치 (Calm-Whisper, arXiv 2505.12969):**
- Whisper-large-v3, UrbanSound8K 비음성 환각률 **99.97%**
- LibriSpeech WER **2.12% (clean) / 4.07% (other)**
- UrbanSound 클립의 55.2%가 " so"로 출력됨 → 영어로 디코딩된 것으로 추정. 논문에 언어 설정이 명시되지 않았으므로 `language="en"`, greedy(temperature 0)로 시작하고 설정을 기록한다.

---

## 6. 실험 조건 (RQ1)

1. 비음성: 환경음, 음악, 잡음 클립만 입력
2. 무음 길이: 음성 앞뒤에 0.5~10초 무음 추가
3. 저SNR: 음성에 잡음을 +20 ~ −5 dB로 혼합
4. 실제 회의 음성: 정답 대비 삽입 오류 분석

**RQ2 계측:** 확산·flow 디코더의 단계별 중간 출력과 토큰 신뢰도를 저장하고, 환각 토큰이 처음 나타난 단계와 이후 유지·삭제 여부를 추적한다.

**RQ3 탐지:** 후보 간 평균 편집거리, 토큰별 투표 일치율을 점수로 쓴다.

---

## 7. 1주차 할 일과 진행/중단 판정 (**가장 먼저**)

### 할 일 (권장 순서)
1. [x] 저장소 뼈대 만들기 (아래 8절) — 2026-10-16 완료. 공통 인터페이스: `src/runners/base.py`
2. [ ] Whisper-large-v3: LibriSpeech test-clean 소량 + UrbanSound 소량으로 구동 확인
3. [ ] Drax: 설치 → 같은 샘플로 구동, 중간 단계 출력을 꺼낼 수 있는지 확인
4. [ ] Whisper-LLaDA: Colab에서 환경 구성 → 체크포인트 로드 → 순수 확산과 semi-AR 각각 1~2클립 구동. 메모리와 클립당 시간 실측
5. [x] Whisper-LLaDA 설정 확인: 인코더 동결 여부, 학습 데이터 (README·config·논문) — 2026-10-16 완료, 3.3절 참고
6. [ ] Whisfusion v1 설치 시도 (실패해도 진행)
7. [ ] 기준 재현: Whisper-large-v3 UrbanSound 환각률(99.97% 근처), LibriSpeech WER(2.12% 근처)
8. [ ] 실측값(메모리, 클립당 시간, 예상 GPU 시간)으로 이 문서와 RP 갱신

### 판정 기준
- **통과:** Drax와 Whisper-LLaDA가 LibriSpeech 샘플에서 정상적인 전사를 출력하고, Whisper 기준 수치가 재현됨
- **실패:** 둘 중 하나라도 1주 안에 구동하지 못하면 → 예비 주제로 전환 (10절)

### Colab 운용 주의
- 모델마다 Colab 세션(노트북)을 분리한다. torch 버전이 서로 충돌한다(Whisfusion 2.4.1 vs Whisper-LLaDA 2.7.1).
- GPU는 A100 40GB 기본. 비용은 2026년 3월 기준 시간당 약 5.4 유닛이고, Pro+는 월 500 유닛이다.
- 중간 결과는 Google Drive에 저장한다. 세션이 끊겨도 이어서 할 수 있게 한다.

---

## 8. 권장 저장소 구조 (뼈대)

```
diffusion-asr-hallucination/
├── CLAUDE.md                 # 이 문서
├── notebooks/
│   ├── 00_whisper_baseline.ipynb
│   ├── 01_drax_smoke.ipynb
│   ├── 02_whisper_llada_smoke.ipynb
│   └── 03_whisfusion_v1_smoke.ipynb
├── src/
│   ├── data/                 # 데이터 로딩, 무음/잡음 합성
│   ├── runners/              # 모델별 추론 래퍼 (공통 인터페이스)
│   ├── metrics/              # HR, IER, WER, 환각 유형 분류
│   └── analysis/             # 궤적 분석, 탐지 AUROC
├── configs/                  # 실험 조건별 설정
└── results/                  # 결과 JSON/CSV (설정 포함)
```

모델별 추론 래퍼는 같은 입력(오디오 경로 목록)을 받아 같은 형식(전사, 후보 목록, 단계별 중간 출력, 설정)을 돌려주게 만든다.

---

## 9. 8주 일정

| 주차 | 내용 |
|---|---|
| 1 | 환경 구축, 기준 재현, **진행/중단 판정** |
| 2~3 | 비교 ① 모델 내 다이얼 (핵심) |
| 4~5 | 비교 ②·③ 모델 간 비교 · 5주차 중간발표 |
| 6~7 | RQ2 궤적 분석, RQ3 탐지기 |
| 8 | 정리, 최종 보고서 |

중간발표에서 보여줄 것: 기준 재현표, 비교 ① 결과표, 같은 잡음 클립에 대한 모델별 출력 데모.

---

## 10. 위험과 예비 주제

| 위험 | 대응 |
|---|---|
| Whisper-LLaDA·Drax 구동 실패 | 예비 주제로 전환 |
| Whisper-LLaDA 속도 | 비음성 약 1,000클립으로 축소, 조건 우선순위화 |
| 학습 데이터 차이 (비교 ②) | 한계로 명시, 결론은 비교 ① 중심 |
| 선점 | 착수 전·중간발표 전 문헌 재검색 |

**예비 주제:** 트랜스듀서(Parakeet-TDT) 환각의 출처 규명. 내부 언어모델(예측망)에서 오는지, 인코더에서 오는지를 분해한다. HALAS 데이터셋(실제 음성 환각 사람 라벨, Parakeet 출력 포함, GitHub DSP-AGH/HALAS, HF MatBar99/HALAS)을 쓴다.

---

## 11. 주요 참고문헌

- Kwon et al., Whisfusion, arXiv 2508.07048 — https://arxiv.org/html/2508.07048
- Navon et al., Drax, arXiv 2510.04162 — https://huggingface.co/aiola/drax-v1
- Wang et al., Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing, ICASSP 2026 — https://github.com/liuzhan22/Diffusion-ASR
- Yen et al., MDM-ASR, Interspeech 2026 (체크포인트 미공개)
- Wang et al., Calm-Whisper, Interspeech 2025 — https://arxiv.org/html/2505.12969v1
- Barański et al., Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio, ICASSP 2025 — https://arxiv.org/abs/2501.11378
- Khurdula et al., Audio-Native Speech Recognition with a Frozen Discrete-Diffusion Language Model, arXiv 2607.13013
