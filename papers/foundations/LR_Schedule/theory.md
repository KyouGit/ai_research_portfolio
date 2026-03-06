# Learning Rate Schedule — Theory Notes

---

## 핵심 한 줄

학습하면서 learning rate를 어떻게 바꿀지 정하는 전략이다.

---

## 왜 필요한가

$$w = w - \eta \nabla L$$

- **η**: 한 번 업데이트할 때 얼마나 크게 움직일지

초반과 후반의 요구가 다르다:

| 시점 | 상태 | 필요한 lr |
|------|------|----------|
| 초반 | loss 큼, 멀리 이동해야 함 | 크게 |
| 후반 | loss 작음, 미세 조정 필요 | 작게 |

고정 lr은 이 두 요구를 동시에 만족할 수 없다. 그래서 schedule이 필요하다.

---

## 1. Constant (고정)

$$\eta_t = \eta_0$$

가장 단순. 항상 같은 lr 사용.

- 단점: 초반 느리거나 후반 overshoot 발생
- 현재: 거의 사용 안 함

---

## 2. Step Decay

$$\eta = \eta_0 \cdot \gamma^k$$

- k: 감소 횟수, γ: 감소 비율 (예: 0.1)

```
epoch  0~30  → lr = 0.1
epoch 30~60  → lr = 0.01
epoch 60~90  → lr = 0.001
```

- 장점: 단순, 효과 있음
- 사용처: 초기 CNN 학습 (VGG, ResNet 등)

---

## 3. Exponential Decay

$$\eta_t = \eta_0 \cdot e^{-kt}$$

연속적으로 감소. step decay보다 부드럽다.

- 단점: 너무 빨리 lr이 작아질 수 있음
- 초반에 급감하고 후반에 완만해짐

---

## 4. Linear Decay

$$\eta_t = \eta_0 \left(1 - \frac{t}{T}\right)$$

- T: 총 step 수

Transformer 등장 이후 많이 사용됨.

- 장점: 안정적, 구현 쉬움
- 단점: 후반 미세 조정이 cosine보다 거칠 수 있음

---

## 5. Cosine Decay

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

- 처음: 빠르게 감소
- 후반: 매우 천천히 감소 → 미세 조정에 유리

특징:
- 후반 lr이 거의 0에 가까워져서 수렴이 정밀하다
- Vision 모델에서 특히 많이 사용
- LLM 학습에도 사용

---

## 6. Warmup (Transformer 이후 핵심)

초기 lr이 크면 gradient 폭발, 학습 불안정이 생긴다.
→ 처음 몇 step 동안 lr을 서서히 올린다.

```
step    0  → lr = 0
step 1000  → lr = 0.001 (peak)
이후 decay 시작
```

최초 도입: **Attention Is All You Need (2017)** 논문

Transformer schedule 수식:

$$lr = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup\_steps}^{-1.5})$$

- warmup 구간: step 증가 → lr 증가
- 이후 구간: step 증가 → lr 감소

---

## 7. Cosine Decay + Warmup (현재 가장 흔함)

```
warmup (lr 0 → peak)
    → cosine decay (peak → η_min)
```

LLM 학습의 사실상 표준 조합.

```python
# 예시 구조
if step < warmup_steps:
    lr = peak_lr * (step / warmup_steps)          # 선형 증가
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = eta_min + 0.5 * (peak_lr - eta_min) * (1 + cos(π * progress))
```

---

## 8. Reduce on Plateau

validation loss 기반 자동 조절.

```
3 epoch 동안 improvement 없음 → lr × 0.1
```

- 장점: 자동 조절, 직관적
- 단점: noisy metric에 민감. LLM처럼 긴 학습에는 부적합

---

## 9. One Cycle Policy (Leslie N. Smith)

```
lr 증가 → lr 감소  (형태: /\)
```

- 장점: 빠른 학습, regularization 효과
- 사용처: Vision 모델, FastAI

---

## 10. Cosine with Restarts (SGDR)

cosine decay 후 lr을 reset하고 다시 감소.

```
cosine decay → lr reset → cosine decay → lr reset → ...
```

- 장점: local minimum 탈출 가능
- 사용처: 일부 vision/NLP 실험

---

## 현업 사용 정리

| 분야 | 스케줄 |
|------|--------|
| Vision (CNN) | Cosine decay, Step decay |
| Transformer (소/중형) | Warmup + Linear decay |
| LLM | Warmup + Cosine decay |

---

## Optimizer와의 관계

Adam/AdamW는 파라미터별 adaptive lr을 적용하지만,
**global lr schedule은 여전히 필요하다.**

- Optimizer: 각 파라미터의 상대적 lr 조절
- LR Schedule: 전체 학습률의 절대 크기 조절

둘은 독립적으로 작동하고, 둘 다 중요하다.

---

## 핵심 수식 정리

| 스케줄 | 수식 |
|--------|------|
| Step decay | η = η₀ · γᵏ |
| Exponential | η_t = η₀ · e^(-kt) |
| Linear | η_t = η₀(1 - t/T) |
| Cosine | η_t = η_min + ½(η_max - η_min)(1 + cos(πt/T)) |
| Transformer | lr = d^(-0.5) · min(step^(-0.5), step · warmup^(-1.5)) |

---

## 핵심 요약 3가지

1. 초반에는 lr을 크게(또는 warmup으로 서서히 올려), 후반에는 작게 줄이는 것이 기본 원칙
2. 현재 LLM 표준은 warmup + cosine decay 조합
3. Optimizer는 상대적 lr 조절, LR Schedule은 절대 크기 조절 — 둘 다 필요하다

---

## 직접 해볼 실습 2가지

1. cosine decay / linear decay / constant lr을 같은 모델에 적용하고 validation loss curve를 비교
2. warmup steps를 100 / 500 / 2000으로 바꾸면서 초반 학습 안정성 차이를 loss curve로 확인

---

## 예상 질문 / 답변

**Q1. warmup이 왜 필요한가?**
초기 파라미터가 랜덤 초기화 상태라 gradient가 불안정하다. 큰 lr로 시작하면 gradient 폭발 위험이 있어서 서서히 올린다.

**Q2. cosine decay가 linear decay보다 좋은 이유는?**
후반에 lr이 매우 천천히 줄어들어서 수렴 직전 미세 조정이 더 정밀하다. linear는 후반에도 일정 속도로 감소해서 너무 거칠 수 있다.

**Q3. Adam을 쓰는데도 lr schedule이 필요한가?**
필요하다. Adam은 파라미터별 상대적 lr을 조절하지만, 전체 lr의 절대 크기는 schedule로 따로 조절해야 한다.

**Q4. Transformer schedule 수식에서 warmup 이후 감소하는 이유는?**
step^(-0.5)에 의해 자동으로 감소한다. warmup 전에는 step이 작아서 두 번째 항(step · warmup^(-1.5))이 작고, warmup 후에는 첫 번째 항(step^(-0.5))이 작아서 감소한다.

**Q5. Reduce on Plateau가 LLM에 안 맞는 이유는?**
LLM 학습은 수백만 step이고, validation loss 측정이 비싸다. 또한 loss가 일시적으로 정체되는 경우가 많아서 불필요한 lr 감소가 자주 발생한다.
