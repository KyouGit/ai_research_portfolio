# Backpropagation (역전파) — Theory Notes

---

## 핵심 한 줄

모델의 오차가 각 파라미터에 얼마나 영향을 주는지 계산하는 방법이다.
즉, **어떤 가중치를 얼마나 바꾸면 오차가 줄어드는지** 계산하는 알고리즘이다.

---

## 문제 설정

신경망은 함수다:

$$y = f(x, w)$$

- **x**: 입력
- **w**: 파라미터 (학습 대상)
- **y**: 예측값

정답 y*와의 차이로 loss를 만든다:

$$L = \text{loss}(y, y^*)$$

목표:

$$\min_w L$$

---

## gradient가 필요한 이유

loss를 줄이려면 파라미터를 어느 방향으로 얼마나 바꿔야 하는지 알아야 한다.

$$\frac{\partial L}{\partial w}$$

이것이 gradient다. 이 값이 양수면 w를 줄여야 loss가 줄고, 음수면 w를 늘려야 한다.

---

## 핵심 아이디어: Chain Rule (체인룰)

신경망은 여러 함수가 중첩된 구조다:

$$L = f_3(f_2(f_1(x)))$$

이때 맨 앞 파라미터의 gradient를 직접 계산하면 매우 복잡하다.
체인룰을 쓰면 뒤에서부터 순서대로 곱해나갈 수 있다:

$$\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dh} \cdot \frac{dh}{dx}$$

- 뒤에서부터 gradient를 계산하고
- 앞으로 전파(propagation)한다
- 이미 계산한 gradient를 재사용한다 → 효율적

이름이 **back**propagation인 이유: loss에서 시작해서 뒤로(입력 방향으로) 전파하기 때문.

---

## 실제 계산 흐름

### 1. Forward Pass

$$x \xrightarrow{\text{layer}_1} z_1 \xrightarrow{\text{layer}_2} z_2 \xrightarrow{\text{layer}_3} y$$

모든 중간 결과(activation)를 저장해둔다. backward에서 재사용하기 위해.

### 2. Loss 계산

$$L = (y - y^*)^2 \quad \text{(MSE 예시)}$$

### 3. Backward Pass

$$\frac{dL}{dW_3} \to \frac{dL}{dW_2} \to \frac{dL}{dW_1}$$

loss에서 시작해서 각 layer의 gradient를 순서대로 계산.
한 번 계산한 gradient를 다음 layer 계산에 재사용한다.

---

## 간단한 예 (단층)

모델: $y = wx$

Loss: $L = (y - y^*)^2 = (wx - y^*)^2$

미분:
$$\frac{\partial L}{\partial w} = 2(wx - y^*) \cdot x$$

이 gradient로 weight 업데이트:
$$w \leftarrow w - \eta \cdot \frac{\partial L}{\partial w}$$

---

## 다층 신경망에서

$$z_1 = W_1 x, \quad z_2 = W_2 z_1, \quad z_3 = W_3 z_2$$

역전파 순서:

$$\frac{dL}{dW_3} = \frac{dL}{dz_3} \cdot z_2^T$$

$$\frac{dL}{dW_2} = \frac{dL}{dz_2} \cdot z_1^T, \quad \frac{dL}{dz_2} = W_3^T \cdot \frac{dL}{dz_3}$$

$$\frac{dL}{dW_1} = \frac{dL}{dz_1} \cdot x^T, \quad \frac{dL}{dz_1} = W_2^T \cdot \frac{dL}{dz_2}$$

**핵심**: 뒤에서 계산한 $\frac{dL}{dz}$를 앞 layer 계산에 재사용 → 전체 비용이 forward pass와 거의 동일.

---

## 계산 효율

역전파의 놀라운 점:
- gradient 계산 비용 ≈ forward pass 비용
- 파라미터 수에 관계없이 2번의 pass로 모든 gradient 계산 가능
- 수십억~수천억 파라미터 모델도 학습 가능한 이유가 여기 있다

---

## 초기 딥러닝에서 생긴 문제

### Vanishing Gradient

gradient가 layer를 거칠수록 점점 작아진다.

발생 원인: sigmoid, tanh의 gradient가 최대 0.25, 0에 가까운 경우가 많다.

$$\sigma'(x) = \sigma(x)(1-\sigma(x)) \leq 0.25$$

레이어가 깊으면:

$$0.25^{10} \approx 10^{-6}$$

앞쪽 레이어는 사실상 학습이 안 된다.

### Exploding Gradient

반대로 gradient가 layer를 거칠수록 폭발적으로 커진다.
→ 학습 발산, NaN 발생.

---

## 해결 기술

### ReLU

$$\text{ReLU}(x) = \max(0, x), \quad \frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}$$

양수 구간에서 gradient = 1. 사라지지 않는다.
딥러닝 혁명의 시작.

### Residual Connection

$$\text{output} = F(x) + x$$

gradient가 skip connection을 통해 직접 흐른다:

$$\frac{dL}{dx} = \frac{dL}{d\text{output}} \cdot (F'(x) + 1)$$

항상 최소 1 이상의 gradient가 흐른다. → 100층 이상 학습 가능 (ResNet, Transformer).

### Normalization

- **Batch Normalization**: 각 batch에서 평균/분산 정규화
- **Layer Normalization**: 각 sample 내에서 정규화 (Transformer 사용)

gradient 크기를 안정화한다.

### Gradient Clipping

gradient norm이 임계값을 넘으면 잘라낸다:

$$g \leftarrow g \cdot \frac{\text{clip\_value}}{\|g\|}$$

Exploding gradient 방지. RNN, LLM 학습에서 자주 사용.

---

## 자동 미분 (Automatic Differentiation)

현대 딥러닝에서는 역전파를 직접 구현하지 않는다.
프레임워크가 **computation graph**를 만들고 자동으로 처리한다.

```python
# PyTorch 예시
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)

y = w * x
L = (y - 5.0) ** 2

L.backward()    # 역전파 자동 실행
print(w.grad)   # dL/dw = 2(wx - y*) * x = 2(6-5) * 2 = 4.0
```

동작 원리:
1. forward 시 각 연산의 gradient 함수를 computation graph에 저장
2. `backward()` 호출 시 graph를 역순으로 traversal하며 gradient 계산

대표 프레임워크: **PyTorch** (define-by-run), **JAX** (functional), **TensorFlow**

---

## 현대 LLM에서의 한계와 해결

### Activation Memory 문제

역전파는 forward의 모든 activation을 저장해야 한다.
LLM에서는 activation memory가 파라미터 메모리보다 클 수 있다.

해결:
- **Gradient Checkpointing**: activation 일부만 저장하고 나머지는 backward 시 재계산
- **Recomputation**: 메모리 대신 연산량으로 트레이드오프

### 분산 학습

GPU 수천 개로 학습 시 backward도 분산해야 한다:
- **Pipeline Parallelism**: layer 단위로 GPU 분산
- **Tensor Parallelism**: 행렬 연산 단위로 GPU 분산
- **Data Parallelism**: 데이터를 나눠서 gradient 평균

---

## 최신 연구 방향

| 방향 | 내용 |
|------|------|
| Forward-Forward Algorithm (Hinton) | backprop 없이 forward pass만으로 학습 |
| Local Learning | 각 layer가 독립적으로 local loss로 학습 |
| Biological Plausibility | 뇌는 backprop처럼 작동하지 않음 → 대안 탐구 |
| Score Matching (Diffusion) | gradient 대신 score function ∇ log p(x)를 학습 |

---

## 핵심 요약 3가지

1. 역전파 = chain rule을 이용해 loss gradient를 뒤에서 앞으로 전파. 계산 비용 ≈ forward pass
2. Vanishing/Exploding gradient 문제 → ReLU, Residual connection, Normalization, Gradient clipping으로 해결
3. PyTorch 등 프레임워크의 autograd가 역전파를 자동 처리. `loss.backward()`가 전부다

---

## 직접 해볼 실습 2가지

1. 2층 MLP를 NumPy만으로 구현하고 역전파 수식을 직접 코딩. PyTorch 결과와 gradient 값 비교
2. sigmoid vs ReLU 활성화 함수를 쓴 10층 네트워크에서 각 layer의 gradient norm을 시각화. vanishing 현상 직접 확인

---

## 예상 질문 / 답변

**Q1. 역전파와 gradient descent의 차이는?**
역전파는 gradient를 계산하는 방법이고, gradient descent는 그 gradient로 파라미터를 업데이트하는 optimizer다. 역할이 다르다.

**Q2. forward pass와 backward pass의 계산 비용 비율은?**
대략 backward가 forward의 2배 정도 (activation 저장 + gradient 계산). 하지만 같은 오더(O)다.

**Q3. gradient checkpointing을 쓰면 뭐가 달라지나?**
activation을 일부만 저장해서 메모리를 줄이는 대신, backward 시 필요한 activation을 재계산한다. 메모리 ↓, 연산량 ↑ 트레이드오프.

**Q4. ReLU가 vanishing gradient를 해결하는 이유는?**
양수 구간에서 gradient = 1로 일정하다. sigmoid처럼 0.25 이하로 줄어들지 않는다. 단, 음수 구간에서는 gradient = 0 (Dead ReLU 문제).

**Q5. Residual connection이 왜 gradient 흐름에 도움이 되나?**
output = F(x) + x 구조에서 dL/dx = dL/d_output · (F'(x) + 1). F'(x)가 0이더라도 항상 1이 더해져 있어서 gradient가 완전히 사라지지 않는다.
