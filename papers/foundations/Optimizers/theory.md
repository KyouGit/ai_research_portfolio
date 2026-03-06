# Optimizer 전체 흐름 — Theory Notes

---

## 옵티마이저란

옵티마이저는 그래디언트를 이용해서 파라미터를 어떻게 업데이트할지 결정하는 알고리즘이다.

```
gradient 계산 → optimizer가 weight 업데이트
```

optimizer가 바뀌면 학습 속도, 안정성, 성능이 달라진다.

---

## 1. SGD (기본)

$$w_{t+1} = w_t - \eta \nabla L(w_t)$$

- **w**: 파라미터
- **η**: learning rate
- **∇L**: gradient (loss의 기울기)

| 특징 | 내용 |
|------|------|
| 장점 | 단순, 계산 빠름 |
| 단점 | 학습 느림, learning rate 민감 |

노이즈가 있어서 local minimum 탈출에 유리하지만, 수렴이 불안정하다.

---

## 2. SGD + Momentum

**아이디어**: 물리학의 관성. 이전 이동 방향을 기억한다.

$$v_t = \beta v_{t-1} + \nabla L$$
$$w = w - \eta v_t$$

- **β**: momentum 계수 (보통 0.9)
- **v_t**: velocity (이전 gradient 누적)

| 특징 | 내용 |
|------|------|
| 장점 | 진동 감소, 수렴 속도 증가 |
| 사용처 | CNN 학습 (ResNet, VGG 등) |

SGD + momentum은 이미지 분류에서 여전히 많이 쓰인다.

---

## 3. AdaGrad

**아이디어**: 파라미터마다 다른 learning rate를 적용한다.

$$w = w - \frac{\eta}{\sqrt{G + \epsilon}} \nabla L$$

- **G**: gradient 제곱 누적합 (지금까지의 모든 gradient²)
- **ε**: 분모가 0이 되지 않도록 하는 작은 값

**직관**: 자주 업데이트된 파라미터는 G가 크므로 learning rate가 작아지고,
드물게 업데이트된 파라미터는 G가 작으므로 learning rate가 크다.

| 특징 | 내용 |
|------|------|
| 장점 | sparse 데이터(NLP 임베딩 등)에 좋다 |
| 단점 | G가 계속 쌓여서 learning rate가 0에 수렴 → 학습 멈춤 |

---

## 4. RMSProp

**아이디어**: AdaGrad의 누적 문제를 moving average로 해결한다.

$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta) g_t^2$$
$$w = w - \frac{\eta}{\sqrt{E[g^2] + \epsilon}} g$$

- **E[g²]_t**: gradient 제곱의 지수이동평균 (오래된 것은 잊는다)
- **β**: decay rate (보통 0.9)

AdaGrad는 과거 gradient를 전부 누적하지만, RMSProp은 최근 것에 더 가중치를 준다.

| 특징 | 내용 |
|------|------|
| 장점 | learning rate 안정적, RNN 학습에 적합 |
| 한계 | momentum이 없음 |

---

## 5. Adam (가장 유명)

**아이디어**: momentum + RMSProp을 합친다.

### 1차 모멘트 (gradient 방향, momentum)
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

### 2차 모멘트 (gradient 크기, RMSProp)
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

### Bias correction (초기 step 보정)
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

### 업데이트
$$w = w - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**기본값**: β₁=0.9, β₂=0.999, ε=1e-8

| 특징 | 내용 |
|------|------|
| 장점 | 학습 빠름, 튜닝 쉬움, 대부분 문제에서 잘 작동 |
| 사용처 | CNN, Transformer, Diffusion 등 범용 |

---

## 6. AdamW (LLM 표준)

**문제**: Adam에서 weight decay가 gradient에 섞여서 잘못 적용됨.

**해결**: weight decay를 gradient update와 분리한다.

$$w = w - \eta \hat{m}_t - \lambda w$$

- **λ**: weight decay 계수 (파라미터를 0 방향으로 당기는 정규화)
- 마지막 항 `-λw`가 gradient와 독립적으로 적용됨

| 특징 | 내용 |
|------|------|
| 장점 | generalization 향상, weight decay 정확히 적용 |
| 사용처 | BERT, GPT, LLaMA 등 대부분 LLM |

---

## 7. LAMB

**문제**: 대형 모델에서 batch size를 매우 크게 쓰면 (예: 64k) Adam이 불안정.

**아이디어**: 레이어별로 gradient norm과 weight norm 비율로 learning rate를 조절.

$$w_l = w_l - \eta \cdot \frac{\|w_l\|}{\|\hat{m}_l\|} \cdot \hat{m}_l$$

| 특징 | 내용 |
|------|------|
| 장점 | large batch에서 안정적 |
| 사용처 | Google BERT 사전학습 |

---

## 8. Lion (최근 연구)

**아이디어**: gradient magnitude 대신 **gradient sign**만 사용.

$$w = w - \eta \cdot \text{sign}(\beta_1 m_{t-1} + (1-\beta_1) g_t)$$

| 특징 | 내용 |
|------|------|
| 장점 | 메모리 절약, LLM 학습 효율 |
| 한계 | Adam보다 hyperparameter 민감, 아직 AdamW가 주류 |

---

## 최신 연구 흐름

| 방향 | 대표 기술 | 설명 |
|------|----------|------|
| 메모리 효율 | 8-bit optimizer (bitsandbytes) | optimizer state를 8bit로 압축 |
| 대형 모델 스케일링 | LAMB, Adafactor | 대규모 batch/모델에서 안정적 학습 |
| Adafactor | Google T5 사용 | 2차 모멘트를 행렬 분해로 근사해 메모리 대폭 절감 |
| optimizer-free | Sign Descent, LISA | optimizer state 없이 학습 가능성 연구 (아직 연구 단계) |

---

## 실제 사용 가이드

| 상황 | 추천 Optimizer |
|------|---------------|
| 작은 모델, 범용 | Adam |
| 이미지 분류 (CNN) | SGD + momentum |
| Transformer / LLM | AdamW |
| 대형 모델 사전학습 (large batch) | AdamW + LAMB, Adafactor |

---

## 중요한 포인트: learning rate schedule이 더 중요하다

실제로 연구자들이 optimizer보다 더 많이 고민하는 것이 learning rate schedule이다.

대표 패턴:

```
warmup (초반 학습률을 천천히 올림)
    → peak lr
        → cosine decay (점점 줄임)
```

이것이 optimizer 선택보다 성능에 훨씬 큰 영향을 준다.

---

## 핵심 요약 3가지

1. optimizer 발전 흐름: SGD → Momentum → AdaGrad → RMSProp → Adam → AdamW → (LAMB, Lion)
2. Adam = momentum(방향) + RMSProp(크기 조절). 두 아이디어의 결합
3. LLM은 AdamW 표준. 하지만 learning rate schedule이 optimizer 선택보다 성능에 더 큰 영향을 준다

---

## 직접 해볼 실습 2가지

1. SGD / Adam / AdamW 세 가지를 같은 모델에 적용하고 loss curve와 최종 accuracy를 비교
2. AdamW에 warmup + cosine decay schedule을 추가하고, 없을 때와 비교해서 수렴 속도 차이 측정

---

## 세부 문서 링크

- [SGD 상세 노트](../SGD/theory.md)
