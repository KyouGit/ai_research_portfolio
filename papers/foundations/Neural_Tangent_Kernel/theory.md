# Neural Tangent Kernel (NTK)

## 한 줄 요약
NTK는 충분히 큰 폭의 신경망에서 학습 동역학이 커널 회귀와 유사해진다는 이론적 프레임이다.

## 문제의식
- 파라미터 수가 데이터 수보다 훨씬 큰데도 왜 학습이 잘 되는가?
- 왜 과매개변수화 구간에서 일반화가 다시 좋아지는 현상이 보이는가?

## 선형화 핵심
신경망 출력이 `f(x, θ)`일 때 작은 업데이트 `Δθ`에 대해:

`f(x, θ + Δθ) ≈ f(x, θ) + ∇_θ f(x, θ) · Δθ`

즉, 학습이 gradient feature 공간에서의 선형 모델처럼 해석된다.

## NTK 정의
두 입력 `x, x'`에 대해:

`K(x, x') = ∇_θ f(x)^T ∇_θ f(x')`

- 의미: 두 입력의 gradient feature 유사도
- 폭이 매우 큰 네트워크에서는 학습 중 `K`가 거의 변하지 않는 근사가 성립

## 중요한 귀결
- gradient descent의 함수 공간 동역학이 kernel regression과 유사해진다.
- overparameterized setting에서 optimization이 쉬워지는 현상을 설명하는 기반을 제공한다.

## 일반화/Double Descent와의 연결
- 전통 관점: 복잡도 증가 -> 과적합 증가
- 현대 딥러닝 관측: `감소 -> 증가(보간 임계점) -> 다시 감소` (double descent)
- 큰 모델에서는 표현력 확장과 SGD의 암묵적 편향(예: flat minima 선호)으로 일반화가 다시 개선될 수 있다.

## 주의사항
- NTK는 infinite-width 근사에서 가장 정확하다.
- 현대 LLM의 feature learning, representation drift를 NTK 하나만으로 완전 설명하긴 어렵다.
- mean-field/implicit bias/optimization theory와 함께 보는 것이 실무적으로 유효하다.

## 실무 연결 포인트
- 대규모 모델 학습이 왜 안정적으로 되는지 이해할 때 유용한 이론적 도구
- 모델 크기/데이터/학습률 스케줄이 만드는 동역학 해석의 출발점

