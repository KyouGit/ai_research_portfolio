# 왜 Backprop이 그렇게 효율적인가 — 계산 복잡도와 Gradient 구조

---

## 핵심 결론

Backprop은 모든 gradient를 **forward 계산 비용의 약 2배** 정도로 계산할 수 있다.
파라미터가 1억 개든 1000억 개든 상관없이.

---

## 1. Naïve 계산 vs Backprop

신경망:

$$y = f(x, w_1, w_2, \ldots, w_n)$$

학습에 필요한 gradient:

$$\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \ldots, \frac{\partial L}{\partial w_n}$$

### Naïve 방법 (수치 미분)

각 파라미터마다 독립적으로 미분 계산:

$$\frac{\partial L}{\partial w_i} \approx \frac{L(w_i + \epsilon) - L(w_i)}{\epsilon}$$

- 계산 횟수 = 파라미터 수 × forward 1회 비용
- 파라미터 1억 개 → **1억 번의 forward pass**
- 완전히 불가능하다

### Backprop 방법

중간 gradient를 재사용한다.

계산 횟수 = O(연산 수) ≈ forward 1회 비용

파라미터가 1억 개 → **forward 2회 비용**

---

## 2. 왜 재사용이 가능한가

신경망은 합성함수 구조다:

$$L = f_3(f_2(f_1(x)))$$

chain rule:

$$\frac{dL}{dx} = \underbrace{\frac{dL}{df_3}}_{\delta_3} \cdot \underbrace{\frac{df_3}{df_2}}_{\text{local}} \cdot \underbrace{\frac{df_2}{df_1}}_{\text{local}} \cdot \underbrace{\frac{df_1}{dx}}_{\text{local}}$$

**핵심**: $\delta_3 = \frac{dL}{df_3}$는 모든 하위 layer의 gradient 계산에 공통으로 들어간다.

```
forward:
x → [f1] → z1 → [f2] → z2 → [f3] → y → loss

backward:
δ3 = dL/dz3                     (1번 계산)
δ2 = δ3 · df3/dz2               (δ3 재사용)
δ1 = δ2 · df2/dz1               (δ2 재사용)

각 layer의 weight gradient:
dL/dW3 = δ3 · z2.T
dL/dW2 = δ2 · z1.T
dL/dW1 = δ1 · x.T
```

각 레이어에서 계산하는 것:
- **local gradient**: 해당 layer의 연산 미분 (forward 때 이미 계산한 값 활용)
- **upstream gradient (δ)**: 위에서 내려온 값 (재사용)

→ 전체 비용 = forward 비용 + backward 비용 ≈ 2 × forward

---

## 3. 복잡도 정리

| 방법 | 계산 비용 |
|------|----------|
| 수치 미분 (naïve) | O(n × forward) — n: 파라미터 수 |
| Backpropagation | O(forward) — 파라미터 수 무관 |

실제로는 backward ≈ 2 × forward 정도.
(activation 저장 오버헤드 포함)

**이것이 딥러닝이 현실에서 가능해진 가장 근본적인 이유다.**

---

## 4. Low Effective Rank Gradient

최근 딥러닝 이론에서 중요한 관찰.

### 현상

파라미터 공간은 수십억 차원이지만,
실제 학습 중 gradient가 움직이는 방향은 훨씬 적다.

```
파라미터 차원: 175B (GPT-3)
실제 gradient rank: 수백 ~ 수천 수준
```

이것을 **low effective rank gradient**라고 한다.

### 왜 이런 현상이 생기나

자연 데이터는 고차원처럼 보이지만 실제로는 **low-dimensional manifold** 위에 있다.

- 이미지: 수백만 픽셀이지만 실제 변화는 몇 가지 방향 (조명, 자세, 배경)
- 텍스트: 수십만 단어지만 의미는 제한된 개념 공간

데이터의 구조가 단순하면 → gradient도 소수 방향으로 집중된다.

### 실제 관찰

gradient의 covariance matrix를 eigen-decomposition 하면:

```
eigenvalue 분포:
상위 몇 개의 eigenvalue >> 나머지 전체
```

즉, gradient 분포의 대부분이 소수 방향으로 집중.

### 실용적 의미

이 관찰이 **LoRA (Low-Rank Adaptation)** 의 이론적 근거다.

```python
# 원래 weight update
ΔW ∈ R^(d × k)   # d × k 크기의 full matrix

# LoRA: low-rank로 근사
ΔW = A @ B        # A: d×r, B: r×k, r << min(d, k)
```

gradient가 실제로 low-rank 공간에서 움직이기 때문에,
low-rank 행렬만 학습해도 full fine-tuning에 가까운 성능이 나온다.

---

## 5. Gradient Noise Scale (OpenAI)

대형 모델 학습에서 **optimal batch size**를 결정하는 개념.

### 왜 noise가 생기나

SGD는 전체 데이터 대신 mini-batch를 사용한다:

$$g_{\text{batch}} = \frac{1}{B}\sum_{i \in \text{batch}} \nabla L_i = g_{\text{true}} + \text{noise}$$

batch가 작을수록 noise가 크고, 클수록 noise가 작다.

### Gradient Noise Scale 정의

$$S = \frac{\text{gradient variance}}{\text{gradient magnitude}^2}$$

직관: gradient signal 대비 noise의 비율.

### 핵심 결과

| batch size B vs noise scale S | 효과 |
|-------------------------------|------|
| B << S | gradient 방향이 noisy. step마다 새로운 정보. 효율적 |
| B ≈ S | **optimal**: 추가 compute가 새로운 정보를 줌 |
| B >> S | gradient가 거의 같음. 추가 compute가 낭비 |

**optimal batch size ≈ noise scale S**

### LLM 학습에서의 활용

```
1. 초반 학습: noise scale 측정
2. noise scale에 맞춰 batch size 결정
3. 학습 진행에 따라 noise scale 변화 → batch size 조정
```

GPT 계열 모델이 이 방식으로 batch size를 결정한다.
compute 낭비 없이 학습 효율을 최대화한다.

---

## 6. 세 가지를 연결해서 보면

딥러닝 학습이 가능한 이유:

| 요소 | 의미 |
|------|------|
| **Backprop 효율** | gradient 계산 ≈ 2 × forward. 수천억 파라미터도 학습 가능 |
| **Low-rank gradient** | 실제 학습 차원 << 파라미터 수. 거대한 모델도 수렴 가능 |
| **Gradient noise 활용** | SGD noise가 generalization에 도움. optimal batch size 존재 |

이 세 가지는 **Scaling Law**와도 연결된다:
- gradient 구조가 단순 → 모델을 키워도 학습 방향이 안정적
- noise scale이 batch 효율을 결정 → large batch training 전략
- 이 모든 것이 합쳐져 "파라미터↑, 데이터↑, compute↑ → 성능↑" 패턴을 만든다

---

## 핵심 요약 3가지

1. Backprop의 핵심은 intermediate gradient 재사용. 파라미터가 n개여도 비용은 O(forward). naïve 방법의 O(n × forward)와 비교하면 혁명적 차이
2. 실제 gradient는 저차원 부분공간(low-rank subspace)에 집중된다. LoRA가 작동하는 이론적 근거
3. Gradient noise scale이 optimal batch size를 결정한다. B >> S이면 compute 낭비, B << S이면 step당 정보 획득이 최대

---

## 직접 해볼 실습 2가지

1. 작은 MLP의 gradient covariance matrix를 계산하고 eigenvalue를 시각화. 상위 몇 개의 eigenvalue가 전체의 몇 %를 차지하는지 확인
2. 같은 모델을 batch size 8 / 64 / 512 / 4096으로 학습하고 step당 loss 감소와 wall-clock 시간을 비교. optimal batch를 직접 찾아보기

---

## 예상 질문 / 답변

**Q1. Backprop이 O(forward)인데 왜 backward가 실제로 2배 느린가?**
activation을 저장하는 메모리 접근 비용과 gradient를 별도로 저장하는 오버헤드가 있다. 하지만 파라미터 수와 무관하게 상수 배수다.

**Q2. Low-rank gradient가 사실이면 full fine-tuning이 낭비 아닌가?**
일정 부분 맞다. LoRA는 이 관찰을 이용해서 r=8~64 정도의 low-rank 행렬만 학습해도 full fine-tuning의 95%+ 성능을 낸다. 메모리와 compute를 크게 절약한다.

**Q3. Gradient noise scale이 학습 중에 변하나?**
변한다. 초반에는 noise scale이 크고(loss landscape가 복잡), 후반에는 작아진다. 그래서 학습 중 batch size를 점진적으로 늘리는 전략(batch size warmup)이 효과적이다.

**Q4. 수치 미분(numerical gradient)을 언제 쓰나?**
gradient check에 쓴다. autograd가 맞게 구현됐는지 검증할 때 수치 미분 결과와 backprop 결과를 비교한다. 학습에는 절대 쓰지 않는다.

**Q5. Scaling Law와 gradient structure의 연결은?**
모델이 커져도 gradient가 low-rank 구조를 유지하고, noise scale이 예측 가능하게 변하기 때문에 파라미터/데이터/compute를 늘릴수록 성능이 power-law로 증가하는 것이 가능하다.
