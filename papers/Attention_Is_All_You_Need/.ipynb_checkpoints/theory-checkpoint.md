# Attention Is All You Need: Transformer를 진짜 이해하기

> 목표: "Transformer가 왜 동작하는지"를 대학생 눈높이로 정확히 설명할 수 있게 만들기

## 1. 논문 이전: 무엇이 문제였나?
2017년 이전의 기계번역 주류는 RNN/LSTM 기반 Seq2Seq였다.  
이 구조는 시간축을 따라 순서대로 계산하므로 아래 한계가 있었다.

- 병렬화가 어렵다: 토큰 `t`를 계산하려면 `t-1` 결과가 필요
- 긴 문맥 전달이 어렵다: 앞쪽 정보가 뒤로 갈수록 약해짐
- 학습 시간이 길다: 시퀀스 길이가 길수록 병목이 커짐

Transformer는 "순환 없이도 문맥을 잘 읽을 수 있다"를 보여주며 이 문제를 정면으로 해결했다.

## 2. Transformer 전체 구조
![Transformer Architecture](results/transformer_architecture_diagram.png)

Transformer는 크게 두 부분이다.

- Encoder: 입력 문장을 문맥이 반영된 표현으로 변환
- Decoder: Encoder 정보를 참고해 출력 문장을 한 토큰씩 생성

각 블록의 기본 구성:

- Multi-Head Self-Attention
- Position-wise Feed Forward Network (FFN)
- Residual Connection + LayerNorm

핵심 포인트는 "Attention으로 토큰 간 관계를 직접 계산한다"는 점이다.

## 3. Self-Attention을 직관으로 이해하기
문장 예시: "The animal didn't cross the street because it was tired."

여기서 `it`이 무엇을 가리키는지 판단하려면 문장 전체를 봐야 한다.  
Self-Attention은 `it` 토큰이 다른 토큰들과의 관련도를 점수로 계산해, 어떤 단어를 더 참고할지 스스로 정한다.

즉, 각 단어가 "문장 전체를 읽고" 자기 표현을 업데이트한다.

## 4. Scaled Dot-Product Attention 수식 해설
![Scaled Dot-Product Attention](results/attention_formula_card.png)

```text
Attention(Q, K, V) = softmax((QK^T) / sqrt(d_k)) V
```

각 기호 의미:

- `Q`(Query): 지금 토큰이 찾고 싶은 정보
- `K`(Key): 각 토큰이 "내가 어떤 정보인지"를 나타내는 키
- `V`(Value): 실제 전달할 정보 벡터

계산 과정:

1. `QK^T`: 유사도 점수(누구를 참고할지)
2. `/ sqrt(d_k)`: 차원이 커질 때 점수가 과도해지는 걸 완화
3. `softmax`: 점수를 확률처럼 정규화
4. `V` 가중합: 중요한 토큰 정보는 크게, 덜 중요한 정보는 작게 반영

왜 scaling이 필요한가?

- 내적 값이 너무 크면 softmax가 포화(saturation)돼 gradient가 작아질 수 있다.
- `sqrt(d_k)`로 나누면 학습이 안정된다.

## 5. Multi-Head Attention이 왜 필요한가?
헤드 1개만 쓰면 "한 관점"으로만 문장을 본다.  
하지만 실제 언어는 동시에 여러 패턴이 존재한다.

- 문법 관계(subject-verb agreement)
- 의미 관계(coreference)
- 거리 기반 관계(가까운 단어 vs 먼 단어)

여러 head를 두면 각각 다른 관계를 병렬로 학습하고, 이를 합쳐 더 풍부한 표현을 얻는다.

## 6. Positional Encoding: 순서 정보는 어디서 오나?
Attention 자체는 집합(set) 연산처럼 순서에 둔감하다.  
그래서 토큰 임베딩에 위치 벡터를 더해 순서를 주입한다.

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

이 방식의 장점:

- 서로 다른 주파수의 sin/cos로 위치를 연속적으로 표현
- 상대 위치 차이를 선형적으로 다루기 쉬움
- 학습 길이보다 긴 시퀀스로 일반화할 가능성

## 7. Residual + LayerNorm이 중요한 이유
Transformer는 블록을 깊게 쌓는다.  
깊은 네트워크는 학습 불안정 문제가 생기기 쉬운데,

- Residual: 입력을 우회 연결해 gradient 흐름 보존
- LayerNorm: 레이어 출력 분포를 안정화

두 요소가 있어 deep stack 학습이 현실적으로 가능해진다.

## 8. Decoder에서의 Masked Attention
디코더는 미래 토큰을 미리 보면 안 된다(autoregressive 조건).  
그래서 causal mask를 사용해 현재 시점보다 미래 위치 attention을 차단한다.

이 규칙 덕분에 학습/추론 조건이 일치하고, 생성 모델로 동작한다.

## 9. 복잡도 관점에서 이해하기
Self-Attention의 시간/메모리 복잡도는 시퀀스 길이 `n`에 대해 대략 `O(n^2)`이다.  
RNN의 순차 병목은 줄였지만, 긴 문장에서는 attention matrix 비용이 커진다.

그래서 이후 연구에서 Longformer, Linformer, FlashAttention 같은 효율화가 등장했다.

## 10. 이번 프로젝트 실험과 연결해서 읽기
우리 실험은 WikiText-2 언어모델 설정이다.

- 관찰 지표 1: `train_loss`
- 관찰 지표 2: `val_perplexity`

perplexity는 "모델이 다음 단어를 얼마나 헷갈려 하는지"를 수치화한다.  
값이 작을수록 예측이 더 자신감 있고 정확하다는 뜻이다.

![Experiment Metric Curve](results/metric_curve_20260303_040009.png)

그래프 해석 팁:

- loss/ppl이 내려가면 학습이 진행 중
- train만 내려가고 val이 안 내려가면 과적합 신호
- 진동이 크면 learning rate, batch size, regularization 점검 필요

## 11. 이 논문의 역사적 의미
Attention Is All You Need는 단순한 모델 제안이 아니라,
"언어모델 아키텍처의 기본 철학"을 바꾼 논문이다.

- BERT, GPT 계열의 공통 기반
- 대규모 사전학습-미세조정 패러다임의 핵심 구조
- 현대 LLM 시대의 출발점

## 12. 이 문서를 읽고 설명할 수 있어야 하는 핵심 3가지
- Transformer가 RNN의 어떤 병목을 어떻게 해결했는지
- Attention 수식 각 항이 실제로 어떤 의미를 갖는지
- 실험 지표(loss/perplexity)를 보고 모델 상태를 해석하는 방법
