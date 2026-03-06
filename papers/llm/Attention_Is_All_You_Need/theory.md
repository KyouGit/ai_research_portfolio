# Attention Is All You Need — Theory Notes

---

## 핵심 한 줄

모델을 크게 만들 때 성능이 계속 안정적으로 올라가기 때문에 Transformer는 Scaling에 유리하다.

---

## 왜 Transformer는 Scaling에 유리한가

다른 구조(CNN, RNN)는 일정 크기 이상에서 성능이 잘 안 올라가거나 학습이 불안정해진다.
Transformer는 그런 문제가 상대적으로 적다.

---

## 기존 구조의 Scaling 한계

### RNN (Recurrent Neural Network)

문장 처리 방식:
```
단어1 → 단어2 → 단어3 → 단어4   (순차 처리)
```

문제:
- 긴 문장에서 gradient가 점점 사라진다 → **Vanishing Gradient**
- 긴 문맥 기억 어려움
- 병렬 계산 불가능 → GPU 활용 낮음
- 결론: 모델을 크게 만들어도 학습 효율이 떨어진다

### CNN (Convolutional Neural Network)

구조:
- local receptive field → 현재 단어에서 주변 단어만 봄
- 멀리 떨어진 단어 관계를 이해하려면 layer를 계속 늘려야 함

문제:
- 구조 복잡, 학습 어려움
- Scaling 비효율

---

## Transformer 구조

**논문**: Attention Is All You Need (Google Brain, 2017)

### 핵심 아이디어

Attention으로 모든 토큰을 동시에 본다.

```
token embedding
      ↓
self-attention
      ↓
feed-forward
      ↓
stack layers
```

---

## Self-Attention의 장점

### 1. 모든 토큰을 동시에 본다

예시 문장: `The cat that chased the mouse is black`

- **Transformer**: cat ↔ chased, mouse ↔ chased, cat ↔ is 관계를 직접 연결
- **RNN**: cat → that → chased → mouse 순차로 가야 함

→ Transformer가 장거리 관계 이해가 훨씬 좋다.

### 2. 완전히 병렬 계산 가능

```
RNN:            Transformer:
token1          token1
  ↓             token2   (동시에 처리)
token2          token3
  ↓             token4
token3
```

→ GPU에서 매우 효율적. LLM 학습(수천~수만 GPU 동시 구동)에 최적화된 구조.

### 3. 모델 크기를 키워도 안정적

레이어 반복 구조:
```
attention → feedforward → attention → feedforward → ...
```

| 모델  | Layer 수   |
|-------|-----------|
| BERT  | 24        |
| GPT-3 | 96        |
| GPT-4 | 100+ (추정) |

layer 24 → 48 → 96 → 120으로 계속 쌓을 수 있다.

---

## 핵심 수식 정리

### Self-Attention 수식

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

기호 의미:
- **Q (Query)**: 현재 토큰이 무엇을 찾는지
- **K (Key)**: 각 토큰이 어떤 정보를 제공하는지
- **V (Value)**: 실제 전달할 정보
- **√d_k**: 내적 값이 커지는 것을 막기 위한 스케일링

의미: 토큰 간 중요도(가중치)를 계산해서, 관련 있는 토큰의 정보를 가중합으로 가져온다.

한계: 계산 복잡도 O(n²) → 토큰 길이가 길어지면 비용 폭발

---

## Scaling과 Transformer

실험적으로 발견된 사실:

| 증가 변수   | 결과           |
|------------|----------------|
| parameter ↑ | power-law 성능 향상 |
| data ↑      | power-law 성능 향상 |
| compute ↑   | power-law 성능 향상 |

→ 이것이 Scaling Law (Kaplan et al., 2020)의 핵심 발견

전부 Transformer 기반:
- OpenAI GPT 시리즈
- Google DeepMind Gemini
- Meta LLaMA
- Anthropic Claude

---

## Transformer의 단점

### Attention 복잡도 문제

$$
O(n^2)
$$

| 토큰 길이 | 계산량 |
|---------|--------|
| 1k      | 1M     |
| 10k     | 100M   |
| 100k    | 10B    |

긴 문서 처리 비용이 폭발적으로 증가한다.

### 후속 연구 방향
- **Linear Attention**: 복잡도를 O(n)으로 낮추는 시도
- **Sparse Attention**: 중요한 토큰만 선택적으로 계산
- **FlashAttention**: 메모리 효율적인 exact attention 구현

---

## 핵심 요약 3가지

1. Transformer는 Self-Attention으로 모든 토큰을 동시에 처리 → 병렬화 가능 → 대규모 GPU 학습에 최적
2. RNN(순차) / CNN(local)과 달리 구조가 단순 반복이라 레이어를 무제한 쌓을 수 있음
3. parameter / data / compute 세 축 모두에서 power-law scaling이 성립 → Scaling Law의 기반

---

## 직접 해볼 실습 2가지

1. 작은 Transformer를 직접 구현하고 layer 수를 2 → 4 → 8로 늘리면서 성능 변화 측정
2. RNN vs Transformer를 같은 task(언어 모델링)에서 비교하고 gradient norm 로그 비교

---

## 예상 질문 / 답변

**Q1. Transformer가 Scaling에 유리한 가장 근본적인 이유는?**
병렬 계산이 가능하고, 레이어 반복 구조가 단순해서 파라미터를 늘려도 학습이 불안정해지지 않기 때문이다.

**Q2. RNN 대비 Self-Attention의 장점은?**
모든 토큰 쌍을 직접 연결하므로 장거리 의존성을 한 번에 포착할 수 있다. RNN은 순차 처리 때문에 vanishing gradient 문제가 있다.

**Q3. Self-Attention의 계산 복잡도가 O(n²)인 이유는?**
n개의 토큰 각각이 다른 n개의 토큰과 attention score를 계산하기 때문이다. 토큰 수의 제곱에 비례한다.

**Q4. Scaling Law와 Transformer의 관계는?**
Transformer는 parameter/data/compute 세 축 모두에서 power-law 성능 향상이 나타나는 유일한 구조였다. 이것이 Scaling Law 연구의 기반이 됐다.

**Q5. Transformer의 O(n²) 문제를 해결하는 방법은?**
FlashAttention(메모리 효율), Sparse Attention(선택적 계산), Linear Attention(O(n) 근사) 등의 변형 연구가 이어지고 있다.
