# Drax · Whisper-LLaDA Colab 설치 가이드와 미확인 항목

> 작성일: 2026-10-16 · 근거: 두 저장소의 README, 소스 코드(main 브랜치), 각 논문 HTML
> 아직 Colab에서 실행해 보지 않았다. 아래 명령은 코드를 읽고 짠 것이며, 실행 후 결과로 갱신한다.

---

## 1. 요약: 연구 설계에 영향을 주는 발견

| # | 발견 | 근거 | 영향 |
|---|---|---|---|
| A | **Whisper-LLaDA는 한 번 확정한 토큰을 다시 마스킹하지 않는다** (`decoding` 모드) | `models/WhisperLLaDA.py` `transcribe()`: 마스크 위치만 채우고(`x[transfer_index] = x0`), 채운 위치는 다시 고치지 않음 | RQ2의 "환각 토큰이 이후 지워지는가?"는 이 모드에서 구조상 **항상 '아니오'**다. 질문을 "몇 번째 단계에서, 어떤 신뢰도로 확정되는가"로 바꿔야 한다. 다시 마스킹하는 경로는 deliberation 모드뿐이다 |
| B | **Drax는 토큰이 단계마다 바뀔 수 있다** | `drax/flow/solver.py`: 매 단계 x₁을 새로 샘플링하고 확률적으로 점프. 소스 분포는 uniform(논문: "a uniform distribution") | RQ2의 "생겼다 지워짐" 추적은 Drax에서만 의미가 있다. A와 대비되는 결과 자체가 발견이 될 수 있다 |
| C | **Whisper-LLaDA는 temperature 0, 신뢰도 기반 리마스킹으로 고정돼 있다** | `generate()`가 `transcribe()`를 temperature 인자 없이(기본 0.0), `remasking='low_confidence'`로 호출 | 같은 입력이면 출력이 항상 같다. **RQ3용 후보 K개를 만들려면 코드 수정이 필요하다** (temperature를 0보다 크게 넘기거나 `remasking='random'`). 1주차에 패치 범위를 정한다 |
| D | **Drax는 항상 확률적이다** | solver의 `categorical()`과 `torch.rand` | 시드를 꼭 기록해야 한다. 대신 후보 K개는 시드만 바꿔 얻을 수 있다 |
| E | **Drax 학습 데이터에 Earnings-22와 LibriSpeech가 포함된다** | 논문: 8개 언어, 약 15K시간. LibriSpeech, MLS, AMI, Earnings-22, VoxPopuli, TED-LIUM, CommonVoice-13 등 | 실제 회의 음성 조건(Earnings-22)이 Drax에게만 in-domain일 수 있다. 사용한 분할이 학습에 들어갔는지 확인하거나, 다른 회의 데이터로 바꿀지 검토한다 |
| F | **Whisper-LLaDA 학습 데이터는 LibriSpeech 960h뿐이다** (속도 변형 0.9/1.1 포함) | 논문 3.2절 | 비음성 입력은 이 모델에게 완전히 학습 분포 밖이다. 비교 ②의 한계에 적는다 |
| G | 두 모델 모두 **Whisper 인코더를 동결**했다 | Whisper-LLaDA: 코드 `requires_grad = False` + 논문 "the Whisper encoder is frozen". Drax: 논문 "We kept the encoder frozen" | "같은 인코더" 전제가 성립한다. 비교 ②의 인코더 쪽 차이는 없다 |
| H | Whisper-LLaDA 출력 길이 상한은 **128토큰**(`gen_len`)이다 | `decode_config.yaml` | "길이 폭주" 환각은 128토큰에서 잘린다. 유형 분류 기준에 반영한다 |
| I | Drax 코드 라이선스는 **CC-BY-NC-4.0**이다 | `pyproject.toml`, README | 인수인계 문서의 "CC-BY-4.0"(모델 카드 기준)과 다르다. 학부 연구라 비상업 조건은 문제없지만, 보고서에는 정확히 적는다 |

---

## 2. Drax 설치 (Colab)

README는 conda + Python 3.10 + torch 2.7.0(cu128)을 권장한다. Colab에서는 conda 없이 pip로 충분하다.

```python
# 1) 저장소 받기
!git clone https://github.com/aiola-lab/drax.git /content/drax

# 2) Colab의 torch 버전 확인. 2.7.x가 아니면 README대로 고정 설치
import torch; print(torch.__version__, torch.version.cuda)
# !pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 3) 패키지 설치 (transformers==4.52.3 으로 고정됨 → Colab 기본 버전을 덮어씀. 노트북 분리 필수)
!pip install -q -e /content/drax
```

> `pip install -e`로 설치한 직후에는 import가 안 될 수 있다. 그러면 런타임 재시작 후 다시 실행한다.

### 구동

```python
from drax import Transcriber
asr = Transcriber(model_path="aiola/drax-v1")
r = asr.transcribe("/path/a.wav", language="en", sampling_steps=32, temperature=1e-2)
print(r[0].transcript)
```

- 기본값: `sampling_steps=16`, `temperature=0.1` (코드 기준). README 예시는 32, 0.01.
- 입력 전처리(모노 변환, 16kHz 리샘플)는 `Transcriber`가 알아서 한다.
- **배치 추론 지원** (경로 목록과 언어 목록을 함께 넘김).

### 중간 단계 꺼내기 (RQ2)

`Transcriber.transcribe()`는 중간 단계를 돌려주지 않지만, 내부 solver에 `return_intermediates=True`가 있다. solver를 직접 부르면 된다 (실행 확인 전 초안):

```python
import torch
torch.manual_seed(0)
steps = 32
x_init, preserve_mask, audio_cache = asr._build_batch(["/path/a.wav"], ["en"], None)
traj = asr.solver.sample(
    x_init=x_init,
    step_size=1 / steps,
    time_grid=torch.linspace(0, 1, steps + 1),  # 기록할 시점들
    return_intermediates=True,
    preserve_mask=preserve_mask,
    temperature=1e-2,
    **audio_cache,
)
# traj: (기록 시점 수, 배치, 길이) 토큰 ID → 단계별로 디코딩
texts = [asr.processor.batch_decode(t, skip_special_tokens=True)[0] for t in traj]
```

토큰별 확률(신뢰도)은 solver가 저장하지 않는다. 필요하면 solver 루프의 `p_1t`를 저장하도록 복사본을 만든다.

---

## 3. Whisper-LLaDA 설치 (Colab)

### conda를 쓰지 않는 이유

- `environment.yaml`은 conda 환경 전체를 고정한 파일이다(env 이름은 `diffasr`. 인수인계 문서의 `whisper-llada`와 다름). CPU용 libtorch, cu11·cu12 라이브러리가 섞여 있어 Colab에서 그대로 재현하기 어렵다.
- **추론 경로에서는 k2를 import하지 않는다** (`inference.py`, `models/WhisperLLaDA.py`, `models/Qformer.py` 확인). k2는 건너뛴다.
- 추론에 실제로 필요한 것: torch, transformers, peft, soundfile, pyyaml, tqdm, numpy.

```python
# 1) 저장소 받기
!git clone https://github.com/liuzhan22/Diffusion-ASR.git /content/Diffusion-ASR

# 2) environment.yaml의 핵심 버전에 맞춤
!pip install -q torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
!pip install -q transformers==4.53.2 peft==0.3.0 soundfile pyyaml tqdm
```

- `peft==0.3.0`은 오래된 버전이지만 **꼭 맞춘다.** 체크포인트를 `strict=False`로 불러오기 때문에, peft 버전이 달라 가중치 이름이 어긋나면 **오류 없이 LoRA 가중치가 빠진 채** 돌아간다(아래 4절 확인 방법).

### 가중치 3개

| 가중치 | 출처 | 크기 |
|---|---|---|
| `openai/whisper-large-v3` | HF | 약 3GB (추정) |
| `GSAI-ML/LLaDA-8B-Instruct` | HF | 약 16GB (8B × fp16 2바이트, 추정) |
| 학습 체크포인트 `whisper-large-v3-llada-checkpoint_best_epoch_143.pth` | [Google Drive](https://drive.google.com/file/d/1btAsCXTHQApnRcl_k5Wfn0zSD08yXHZh/view) | 473MB (실측) |

- 매번 다시 받지 않도록 HF 캐시를 Drive에 둔다: `os.environ["HF_HOME"] = "/content/drive/MyDrive/hf_cache"` (모델 import 전에 설정).
- 체크포인트는 Drive 웹에서 "바로가기 추가"로 내 Drive에 넣거나 `gdown`으로 받는다.

### 설정 파일 고치기 (`configs/decode_config.yaml`)

**기본값이 `mode: "semi_ar_deliberation"`이다.** 이 모드는 다른 인식기의 전사(`origin_transcripts`)가 있어야 하므로 그대로 돌리면 실패한다. 반드시 바꾼다.

```yaml
model:
  whisper_path: "openai/whisper-large-v3"        # HF 이름 또는 로컬 경로
  llada_path: "GSAI-ML/LLaDA-8B-Instruct"
  ckpt: "/content/drive/MyDrive/.../whisper-large-v3-llada-checkpoint_best_epoch_143.pth"
data:
  test_data: "/content/drive/MyDrive/.../smoke.json"
  output_path: "/content/drive/MyDrive/.../results/llada_smoke.json"
decode:
  mode: "decoding"
  steps: 128
  block_length: 128   # 순수 확산. semi-AR은 64 또는 32
```

- 제약: `gen_len % block_length == 0`, 그리고 `steps % (gen_len / block_length) == 0`. 위반하면 assert로 멈춘다.

### 입력 데이터 형식

```json
{"annotation": [{"path": "/content/.../a.wav", "text": "정답 문장 (비음성은 빈 문자열)"}]}
```

**`inference.py`는 리샘플·모노 변환을 하지 않는다** (학습용 `dataset.py`만 모노 변환과 1초 미만 패딩을 함). UrbanSound8K·ESC-50은 44.1kHz 스테레오가 섞여 있으므로 미리 **16kHz 모노 wav로 변환**해 둔다. 학습 조건에 맞추려면 1초 미만 클립도 1초로 패딩한다. 이 처리는 `src/data/`에 공통 함수로 만든다.

### 구동

```python
%cd /content/Diffusion-ASR
!python3 inference.py --cfg-path configs/decode_config.yaml
```

- 출력 JSON 필드: `path`, `text`, `llada_prediction`.
- 배치 추론 미지원 (README). 클립을 하나씩 처리한다.

---

## 4. 1주차에 Colab에서 직접 확인할 항목

| 항목 | 확인 방법 |
|---|---|
| Whisper-LLaDA 체크포인트 로드가 정상인가 | `model.load_state_dict(..., strict=False)`의 반환값에서 `unexpected_keys`가 비어 있고, `missing_keys`에 `lora`·`adapter`·`linear_proj`가 없는지 출력해 본다 |
| Whisper-LLaDA GPU 메모리 | `torch.cuda.max_memory_allocated()`. 코드상 LLaDA는 fp16, Whisper 인코더는 fp32로 올라간다 → 가중치만 약 19GB (추정). A100 40GB는 여유, L4 24GB는 빠듯할 수 있음 |
| 클립당 시간 | 순수 확산(128블록)과 semi-AR(64, 32)을 각각 측정 |
| Drax GPU 메모리·클립당 시간 | 같은 방법 |
| Drax 중간 단계 추출 | 2절 코드가 실제로 돌아가는지 |
| Colab torch 버전 | Drax는 2.7.0, Whisper-LLaDA는 2.7.1 기준. 노트북을 분리했으므로 각자 설치 |

## 5. 아직 문서로도 확인 못 한 항목

- Drax 모델 카드(HF)와 코드의 라이선스 차이 (CC-BY-4.0 vs CC-BY-NC-4.0)
- Drax가 학습에 쓴 Earnings-22 분할 (test 분할 포함 여부)
- Whisper-LLaDA 논문의 학습 GPU 사양 (논문에 없음)
