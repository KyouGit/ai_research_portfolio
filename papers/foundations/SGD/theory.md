# SGD (Stochastic Gradient Descent) — Theory Notes

---

## 핵심 한 줄

오차(loss)를 줄이기 위해 그래디언트 방향으로 조금씩 파라미터를 움직이는 방법이다.

---

## 왜 필요한가

신경망 학습의 목표는 loss를 최소로 만드는 파라미터를 찾는 것이다.

```
입력 → 신경망 → 예측
                 ↓
            정답과 비교 → loss 발생
```

예:
- 정답: 고양이 / 예측: 개 → loss 큼
- 정답: 고양이 / 예측: 고양이 → loss 작음

수식으로 표현하면:

$$
\min_w L(w)
$$

- **w**: 모델 파라미터
- **L**: loss function

---

## 기본 Gradient Descent

$$
w = w - \eta \nabla L(w)
$$

- **η (eta)**: learning rate — 한 번에 얼마나 움직일지
- **∇L(w)**: gradient — loss가 어느 방향으로 커지는지

동작:
1. gradient 계산 (loss가 어느 방향으로 올라가는지)
2. 반대 방향으로 이동 (loss를 내리는 방향)

문제: 전체 데이터로 gradient를 계산해야 한다.
- LLM 학습 기준 1조 토큰 → 계산량 엄청 큼, GPU 점유 오래됨

---

## SGD 아이디어

전체 데이터 대신 랜덤하게 일부만 사용한다.

| 방식 | 데이터 수 |
|------|----------|
| Gradient Descent | 전체 (예: 1,000,000) |
| SGD | 일부 (예: 1개 또는 mini-batch) |

이름의 **Stochastic(확률적)** = 매 스텝마다 랜덤 샘플을 뽑는다는 뜻

---

## SGD 업데이트 수식

$$
w = w - \eta \nabla L_i(w)
$$

- **L_i**: 일부 데이터(mini-batch i)로 계산한 loss

전체 데이터 gradient 대신 mini-batch gradient로 근사한다.

---

## 실제 학습 흐름 (Mini-batch SGD)

```
dataset = 1,000,000 samples
batch_size = 128

1. 128개 데이터 → gradient 계산 → weight 업데이트
2. 다음 128개 → gradient 계산 → weight 업데이트
3. 반복 (전체 데이터를 한 번 다 돌면 1 epoch)
```

PyTorch 코드:

```python
for x, y in dataloader:          # mini-batch 로드
    pred = model(x)               # 순전파
    loss = criterion(pred, y)     # loss 계산

    optimizer.zero_grad()         # gradient 초기화
    loss.backward()               # 역전파 (gradient 계산)
    optimizer.step()              # 파라미터 업데이트 (SGD)
```

---

## 특징

### 장점
- **계산이 빠르다**: 전체 데이터 대신 일부만 쓰기 때문
- **노이즈가 도움이 된다**: gradient가 정확하지 않아 생기는 노이즈가
  - local minimum 탈출을 돕는다
  - generalization(일반화 성능)을 높인다

### 단점
- **loss가 들쭉날쭉**: noisy gradient 때문에 수렴이 불안정할 수 있음
- **learning rate 조절이 중요**: 너무 크면 발산, 너무 작으면 느림

---

## 핵심 수식 정리

### ① Gradient Descent
$$
w = w - \eta \nabla L(w)
$$
전체 데이터 기준. 정확하지만 느리다.

### ② SGD
$$
w = w - \eta \nabla L_i(w)
$$
mini-batch 기준. 빠르고 노이즈가 있다.

### ③ 최적화 목표
$$
\min_w L(w) = \min_w \frac{1}{N} \sum_{i=1}^{N} \ell(f(x_i; w), y_i)
$$
- N: 전체 데이터 수
- ℓ: 개별 샘플의 loss
- f(x_i; w): 파라미터 w를 가진 모델의 예측

---

## SGD vs 개선된 Optimizer

| Optimizer | 특징 | 주요 사용처 |
|-----------|------|------------|
| SGD | 기본, momentum 옵션 있음 | CNN, ResNet |
| RMSProp | 학습률 자동 조절 | RNN |
| Adam | SGD + momentum + RMSProp | 범용 |
| AdamW | Adam + weight decay 분리 | LLM (거의 표준) |

**중요**: 모든 optimizer의 기반이 SGD + gradient다. Adam도 결국 gradient를 이용해 파라미터를 업데이트한다.

---

## 핵심 요약 3가지

1. SGD는 전체 데이터 대신 mini-batch로 gradient를 근사해서 빠르게 학습한다
2. noisy gradient는 단점이기도 하지만 local minimum 탈출과 generalization에 도움이 된다
3. 현대 LLM은 AdamW를 쓰지만, 모든 optimizer의 기반 개념은 SGD와 같다

---

## 직접 해볼 실습 2가지

1. SGD와 Adam을 같은 모델/데이터에서 비교하고 loss curve 차이를 시각화
2. learning rate를 크게/작게 바꾸면서 수렴 속도와 안정성 변화 측정

---

## 예상 질문 / 답변

**Q1. SGD가 GD보다 느린데도 왜 쓰나?**
GD는 전체 데이터를 한 번 처리해야 1 step을 할 수 있다. SGD는 mini-batch 하나로 즉시 업데이트하므로, 실제로는 SGD가 훨씬 빠르다.

**Q2. Stochastic이 왜 유리한가?**
노이즈가 local minimum에 빠지는 것을 방지하고, 더 넓은 parameter space를 탐색하게 해서 generalization에 유리하다.

**Q3. learning rate를 크게 하면 어떻게 되나?**
gradient 방향으로 너무 크게 이동해서 loss가 수렴하지 않고 발산할 수 있다.

**Q4. Adam이 SGD보다 항상 좋은가?**
항상은 아니다. 이미지 분류 같은 작업에서는 SGD+momentum이 Adam보다 generalization이 더 좋은 경우가 있다. LLM 학습에서는 AdamW가 표준이다.

**Q5. zero_grad()를 왜 매 iteration마다 해야 하나?**
PyTorch는 기본적으로 gradient를 누적한다. 이전 batch의 gradient가 남아있으면 잘못된 방향으로 업데이트되므로, 매 step 전에 초기화해야 한다.
