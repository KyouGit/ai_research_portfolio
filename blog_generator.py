from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class BlogDraft:
    title: str
    tags: list[str]
    html: str


class BlogGenerator:
    def __init__(self, claude_api_key: str = "") -> None:
        self.claude_api_key = claude_api_key

    def generate(
        self,
        paper_name: str,
        score: float,
        image_path: Path,
        code_snippet: str,
        experiment_context: dict[str, str] | None = None,
        comparison_note: str = "기본 실험 대비 안정적 수렴 확인",
    ) -> BlogDraft:
        title = f"[{paper_name}] 실험 자동화 리포트"
        ctx = experiment_context or {}
        category = ctx.get("category", "General")
        tags = ["AI", "논문리뷰", "실험자동화", category, paper_name.replace(" ", "_")]
        is_transformer = "transformer" in paper_name.lower() or "attention is all you need" in paper_name.lower()
        task_type = ctx.get("task_type", "next_token_prediction")

        transformer_explain = ""
        if is_transformer:
            transformer_explain = """
<h3>Transformer 필수 개념 정리</h3>
<p><strong>Self-Attention</strong>은 각 단어가 문장 내 다른 모든 단어를 참고해 자신의 표현을 업데이트하는 방식입니다. 즉, \"지금 이 단어를 이해할 때 누구를 얼마나 참고할지\"를 직접 계산합니다.</p>
<p><strong>Scaled Dot-Product Attention</strong>은 <code>softmax((QK^T)/sqrt(d_k))V</code> 수식으로 계산합니다. 여기서 Q는 찾고 싶은 정보, K는 비교 기준, V는 실제 전달 정보입니다. <code>sqrt(d_k)</code>로 나누는 이유는 내적값이 너무 커져 학습이 불안정해지는 것을 막기 위함입니다.</p>
<p><strong>Multi-Head Attention</strong>은 한 관점이 아니라 여러 관점으로 문장을 동시에 읽게 해, 문법 관계/의미 관계/거리 관계를 병렬로 학습합니다. 그래서 단일 헤드보다 표현력이 훨씬 좋아집니다.</p>
<p><strong>Positional Encoding</strong>은 Attention이 단어 순서를 직접 알지 못한다는 약점을 보완합니다. 토큰 임베딩에 위치 벡터를 더해 \"앞뒤 순서\" 정보를 주입합니다.</p>
<p><strong>Encoder-Decoder</strong> 흐름은 Encoder가 입력 의미를 압축하고, Decoder가 그 정보를 보며 다음 토큰을 예측하는 구조입니다. Decoder는 미래 토큰을 미리 보지 못하도록 마스크를 적용합니다.</p>
""".strip()
        rag_method_explain = ""
        if task_type == "rag_qa":
            rag_method_explain = """
<h3>검색 방법 설명: TF-IDF vs LSA(Dense)</h3>
<p><strong>TF-IDF</strong>는 단어 빈도 기반 희소 벡터 검색입니다. 질문과 문서에 같은 단어가 많이 겹칠수록 점수가 올라갑니다. 구현이 간단하고 빠르지만, 표현이 조금만 바뀌어도 매칭이 약해질 수 있습니다.</p>
<p><strong>LSA(Dense)</strong>는 TF-IDF 행렬을 저차원 의미 공간으로 압축해 밀집 임베딩으로 비교하는 방식입니다. 단어가 완전히 같지 않아도 의미가 비슷하면 더 잘 찾을 수 있습니다.</p>
<p>이번 실험의 핵심은 \"정확히 같은 단어 매칭\"(TF-IDF)과 \"의미 기반 매칭\"(LSA Dense) 중 어떤 방식이 질문-근거 문서 연결에 더 유리한지 확인하는 것입니다.</p>
""".strip()
        formula_block = """
<h3>핵심 수식 정리</h3>
<ul>
  <li><code>Attention(Q,K,V)=softmax((QK^T)/sqrt(d_k))V</code>: 토큰 간 관련도를 확률화해 문맥 벡터를 만드는 핵심 식</li>
  <li><code>Perplexity = exp(CrossEntropy)</code>: 모델의 혼란도를 나타내는 지표(낮을수록 좋음)</li>
  <li><code>score = 1/(1+perplexity)</code>: 실험 비교를 위한 요약 점수(클수록 좋음)</li>
</ul>
<p>수식 해석은 반드시 수치 변화와 연결해서 읽어야 하며, 단순 암기보다 \"왜 이 식이 필요한가\"를 설명할 수 있어야 합니다.</p>
""".strip()

        dataset_name = ctx.get("dataset", "WikiText-2 (프로젝트 기본 설정)")
        iterations = ctx.get("iterations", "-")
        model_name = ctx.get("model", "TransformerLM")
        seed_value = ctx.get("seed", "42")
        ppl_text = ctx.get("final_val_perplexity", ctx.get("perplexity", "-"))
        uni_ppl = ctx.get("baseline_unigram_ppl", "-")
        bi_ppl = ctx.get("baseline_bigram_ppl", "-")
        ppl_gain = ctx.get("ppl_improvement_vs_best_baseline", "-")
        rag_base_retr = ctx.get("baseline_retrieval_acc", "-")
        rag_base_f1 = ctx.get("baseline_answer_keyword_f1", "-")
        rag_base_score = ctx.get("baseline_score", "-")
        rag_retr = ctx.get("transformer_retrieval_acc", "-")
        rag_f1 = ctx.get("transformer_answer_keyword_f1", "-")
        rag_score = ctx.get("transformer_score", "-")
        rag_gain = ctx.get("score_gain_vs_baseline", "-")
        mlm_base_top1 = ctx.get("baseline_top1_acc", "-")
        mlm_base_top5 = ctx.get("baseline_top5_acc", "-")
        mlm_top1 = ctx.get("bert_top1_acc", "-")
        mlm_top5 = ctx.get("bert_top5_acc", "-")
        learned = ctx.get(
            "learned_lessons",
            "Attention 기반 구조가 문맥 정보를 반영하면서 지표를 안정적으로 개선할 수 있음을 확인했다.",
        )
        next_actions = ctx.get(
            "next_actions",
            "학습률/시드를 바꿔 재실험하고, 결과 분산을 함께 비교한다.",
        )
        def _to_float(x: str) -> float | None:
            try:
                return float(x)
            except Exception:
                return None

        uni_f = _to_float(uni_ppl)
        bi_f = _to_float(bi_ppl)
        trans_f = _to_float(ppl_text)
        baseline_candidates = [v for v in [uni_f, bi_f] if v is not None]
        best_baseline = min(baseline_candidates) if baseline_candidates else None

        headline = "Transformer 실험은 n-gram 기반 베이스라인과 비교해 문맥 처리 능력에서 구조적 이점을 보이며, perplexity 개선 가능성을 확인했습니다."
        comparison_summary = "이 비교의 핵심은 \"문맥 창이 넓어질수록 모델의 다음 단어 예측이 안정적으로 개선되는가\"를 보는 것입니다."
        if task_type == "rag_qa":
            headline = "이번 실험은 Transformer 개념 문서를 대상으로 한 RAG QA 시나리오에서, 단순 검색 대비 답변 품질 개선 가능성을 점검했습니다."
            comparison_summary = "핵심은 '질문에 맞는 근거를 정확히 찾고, 근거 기반 답변을 생성했는가'입니다."
        elif task_type == "mlm":
            headline = "이번 실험은 BERT의 핵심인 MLM(마스크 복원)에서 단순 빈도 기반 baseline 대비 정확도 개선을 확인하는 데 초점을 맞췄습니다."
            comparison_summary = "핵심은 '양방향 문맥을 보는 복원 방식이 실제 토큰 예측 정확도를 얼마나 높이는가'입니다."
        elif best_baseline is not None and trans_f is not None:
            if trans_f <= best_baseline:
                headline = "이번 설정에서는 Transformer가 baseline보다 더 낮은 perplexity를 보여 문맥 학습 이점을 수치로 확인했습니다."
                comparison_summary = "현재 실험에서는 baseline 대비 성능 개선이 관찰되어, Attention 기반 구조의 이점이 지표로 이어졌습니다."
            else:
                headline = "이번 실행(저 step/소형 모델)에서는 Transformer perplexity가 baseline보다 높아 아직 수렴이 부족한 상태입니다."
                comparison_summary = "현재 결과는 과소학습 구간에 가깝고, step 확대/모델 확장/튜닝으로 역전 가능성을 검증해야 합니다."

        example_blocks: list[str] = []
        for i in range(1, 6):
            ex_input = ctx.get(f"example_{i}_input", "").strip()
            ex_topk = ctx.get(f"example_{i}_topk", "").strip()
            ex_target = ctx.get(f"example_{i}_target", "").strip()
            ex_type = ctx.get(f"example_{i}_type", "").strip() or "prediction"
            ex_output = ctx.get(f"example_{i}_output", "").strip()
            ex_hit = ctx.get(f"example_{i}_evidence_hit", "").strip()
            if not ex_input and not ex_topk:
                continue
            if ex_type == "rag_qa":
                example_blocks.append(
                    f"<p><strong>예시 {i} (rag_qa)</strong><br/>"
                    f"질문: <code>{ex_input}</code><br/>"
                    f"정답(참조): <code>{ex_target}</code><br/>"
                    f"모델 답변: <code>{ex_output}</code><br/>"
                    f"검색 Top-k(문서/점수): <code>{ex_topk}</code><br/>"
                    f"근거 문서 hit: <code>{ex_hit}</code></p>"
                )
            elif ex_type == "mlm":
                example_blocks.append(
                    f"<p><strong>예시 {i} (mlm)</strong><br/>"
                    f"마스킹 문장: <code>{ex_input}</code><br/>"
                    f"정답 토큰: <code>{ex_target}</code><br/>"
                    f"Top-k 복원 후보: <code>{ex_topk}</code></p>"
                )
            else:
                example_blocks.append(
                    f"<p><strong>예시 {i} ({ex_type})</strong><br/>"
                    f"입력 문맥: <code>{ex_input}</code><br/>"
                    f"정답 토큰: <code>{ex_target}</code><br/>"
                    f"Top-k 예측(확률/순위): <code>{ex_topk}</code></p>"
                )
        examples_html = ""
        if example_blocks:
            examples_html = "<h3>7-1. 실험 결과 예시 (모델 출력 샘플)</h3>" + "".join(example_blocks)

        metric_html = ""
        compare_html = ""
        if task_type == "rag_qa":
            metric_html = f"""
<h3>5. 평가 지표 정의</h3>
<ul>
  <li><strong>retrieval_acc</strong>: 정답 근거 문서를 top-1에서 맞춘 비율</li>
  <li><strong>answer_keyword_f1</strong>: 답변에 핵심 키워드가 포함된 정도</li>
  <li><strong>score</strong>: <code>0.6*retrieval_acc + 0.4*answer_keyword_f1</code></li>
</ul>
<p>이번 실행의 transformer_score는 <strong>{rag_score}</strong> 입니다.</p>
""".strip()
            compare_html = f"""
<h3>6. 다른 모델과 비교 (Baseline vs Transformer-style RAG)</h3>
<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th>방법</th>
      <th>retrieval_acc</th>
      <th>answer_keyword_f1</th>
      <th>overall_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>baseline_tfidf</td>
      <td>{rag_base_retr}</td>
      <td>{rag_base_f1}</td>
      <td>{rag_base_score}</td>
    </tr>
    <tr>
      <td>transformer_style_rag</td>
      <td>{rag_retr}</td>
      <td>{rag_f1}</td>
      <td>{rag_score}</td>
    </tr>
  </tbody>
</table>
<p>score_gain_vs_baseline: <strong>{rag_gain}</strong></p>
<p>{comparison_summary}</p>
""".strip()
        elif task_type == "mlm":
            metric_html = f"""
<h3>5. 평가 지표 정의</h3>
<ul>
  <li><strong>top1_acc</strong>: 정답 토큰을 1순위로 맞춘 비율</li>
  <li><strong>top5_acc</strong>: 정답 토큰이 상위 5개 후보에 포함된 비율</li>
  <li><strong>score</strong>: <code>0.7*top1_acc + 0.3*top5_acc</code></li>
</ul>
<p>이번 실행의 score는 <strong>{score:.4f}</strong> 입니다.</p>
""".strip()
            compare_html = f"""
<h3>6. 다른 모델과 비교 (Baseline vs BERT-style MLM)</h3>
<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th>방법</th>
      <th>top1_acc</th>
      <th>top5_acc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>baseline_unigram</td>
      <td>{mlm_base_top1}</td>
      <td>{mlm_base_top5}</td>
    </tr>
    <tr>
      <td>bert_style_mlm</td>
      <td>{mlm_top1}</td>
      <td>{mlm_top5}</td>
    </tr>
  </tbody>
</table>
<p>score_gain_vs_baseline: <strong>{rag_gain}</strong></p>
<p>{comparison_summary}</p>
""".strip()
        else:
            metric_html = f"""
<h3>5. 평가 지표 정의</h3>
<ul>
  <li><strong>score</strong>: 실험 결과를 한눈에 비교하기 위한 요약 지표</li>
  <li><strong>loss</strong>: 모델 예측 오차(작을수록 좋음)</li>
  <li><strong>perplexity</strong>: 언어모델의 혼란도(작을수록 좋음)</li>
</ul>
<p>이번 실행에서 확인된 score는 <strong>{score:.4f}</strong>, perplexity는 <strong>{ppl_text}</strong> 입니다.</p>
<p>특히 perplexity는 \"모델이 다음 단어를 얼마나 헷갈리는가\"를 나타내므로, 대학생 관점에서는 \"정답 확률 분포를 얼마나 날카롭게 학습했는가\"로 이해하면 됩니다.</p>
""".strip()
            compare_html = f"""
<h3>6. 다른 모델과 비교 (Baseline vs Transformer)</h3>
<table border="1" cellpadding="8" cellspacing="0">
  <thead>
    <tr>
      <th>모델</th>
      <th>핵심 아이디어</th>
      <th>Perplexity(낮을수록 좋음)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Unigram</td>
      <td>이전 문맥 무시, 단어 빈도만 사용</td>
      <td>{uni_ppl}</td>
    </tr>
    <tr>
      <td>Bigram</td>
      <td>직전 1개 단어만 조건부로 사용</td>
      <td>{bi_ppl}</td>
    </tr>
    <tr>
      <td>Transformer</td>
      <td>Self-Attention으로 긴 문맥 반영</td>
      <td>{ppl_text}</td>
    </tr>
  </tbody>
</table>
<p>Baseline 대비 개선률(최고 baseline 기준): <strong>{ppl_gain}</strong></p>
<p>{comparison_summary}</p>
""".strip()
        qa_block = f"""
<h3>예상 질문과 답변</h3>
<p><strong>Q1. 왜 perplexity를 핵심 지표로 봤나요?</strong><br/>
A1. 언어모델에서 perplexity는 \"다음 단어를 얼마나 자신 있게 맞추는지\"를 직접적으로 보여주기 때문에, 구조 비교에 가장 해석이 명확합니다.</p>
<p><strong>Q2. 이번 결과에서 Transformer가 baseline보다 불리한 이유는?</strong><br/>
A2. 현재 설정은 step 수와 모델 크기가 작아 과소학습일 가능성이 큽니다. Transformer는 충분한 학습량에서 장점이 드러나는 경우가 많습니다.</p>
<p><strong>Q3. baseline 비교에서 unigram/bigram을 넣은 이유는?</strong><br/>
A3. 문맥 정보를 거의 못 쓰는 모델과의 차이를 수치로 보여주면 Transformer의 구조적 의미를 더 명확히 설명할 수 있기 때문입니다.</p>
<p><strong>Q4. 다음 실험에서 무엇을 먼저 바꿔야 하나요?</strong><br/>
A4. step 수 확대, learning rate 튜닝, seed 반복 평균 순서로 진행하는 것이 가장 재현성과 해석력을 높입니다.</p>
<p><strong>Q5. 이 실험에서 얻은 핵심 인사이트 한 줄은?</strong><br/>
A5. \"Transformer는 잠재력이 높지만, 작은 학습 예산에서는 baseline보다 늦게 성능이 올라올 수 있다\"는 점입니다.</p>
""".strip()

        html = f"""
<h2>{title}</h2>
<p>이 글은 <strong>논문 개념 설명 + 실험 결과 해석</strong>만 빠르게 볼 수 있도록 구성했습니다.</p>
<p><strong>한눈에 보기:</strong> 1) 왜 필요한가 2) 핵심 개념 3) 실험 그래프 해석 4) 다음 액션</p>
<p><strong>요약 결론:</strong> {headline}</p>

<h3>1. 문제 정의</h3>
<p>기존 순차 모델은 긴 문맥 학습과 병렬화에서 한계가 있습니다. 이번 실험은 {paper_name}의 핵심 아이디어를 재현 가능한 코드로 구현하고, 지표를 통해 왜 이 접근이 효과적인지 이해하는 데 목적이 있습니다.</p>

<h3>2. 핵심 아이디어</h3>
<ul>
  <li>배경 문제: 긴 의존성 학습과 계산 병렬화의 어려움</li>
  <li>해결 방식: Attention 기반 처리로 문맥 반영과 병렬 계산 강화</li>
  <li>장단점: 성능/확장성은 좋지만 계산량과 메모리 부담은 커질 수 있음</li>
</ul>
<p>핵심은 \"순차적으로 한 칸씩 처리\"하던 구조를 \"전체 문맥을 한 번에 바라보는 구조\"로 바꾼 것입니다. 그래서 학습 속도와 문맥 이해력이 동시에 개선됩니다.</p>
{transformer_explain}
{rag_method_explain}
{formula_block}

<h3>3. 실험 목적</h3>
<p>이번 실험은 \"{paper_name} 개념이 실제 지표 개선으로 이어지는가\"를 확인하는 데 목적이 있습니다.</p>

<h3>4. 실험 설정</h3>
<ul>
  <li>데이터셋: {dataset_name}</li>
  <li>모델: {model_name}</li>
  <li>반복 횟수: {iterations}</li>
  <li>랜덤 시드: {seed_value}</li>
  <li>태스크 유형: {task_type}</li>
</ul>
<p>설정 의도: 복잡한 튜닝보다 개념 검증에 초점을 두고, 작은 모델 크기에서 수렴 패턴과 지표 변화를 먼저 확인했습니다.</p>

{metric_html}
{compare_html}

<h3>7. 실험 결과 그래프 해석</h3>
<p>최종 점수는 <strong>{score:.4f}</strong> 입니다. 아래 그래프는 반복 학습 중 성능 추세를 보여줍니다.</p>
<p><img src=\"{image_path}\" alt=\"{paper_name} result graph\" /></p>
<p>그래프에서 손실 감소 또는 지표 수렴이 보이면 모델이 안정적으로 학습하고 있다는 신호입니다. 변동이 크면 학습률/배치 크기/정규화 전략을 점검해야 합니다.</p>
<p>특히 대학생 관점에서 볼 때 중요한 포인트는 \"숫자가 왜 좋아졌는지\"를 구조와 연결해 설명할 수 있는지입니다. 예를 들어 perplexity가 감소했다면, 모델이 다음 토큰 분포를 더 날카롭고 정확하게 맞추고 있다는 뜻입니다.</p>
<p>추가 해석: 초반 구간에서 급격히 지표가 개선되고, 후반으로 갈수록 완만해지는 패턴은 \"기초 패턴 학습 -> 미세 조정\" 단계로 전환되는 전형적인 학습 곡선과 유사합니다.</p>
{examples_html}

<h3>8. 기존 방법과 비교</h3>
<p>{comparison_note}</p>

<h3>9. 인사이트 (왜 이런 결과가 나왔는가)</h3>
<ul>
  <li>Self-Attention은 토큰 간 직접 연결을 만들어 장거리 의존성을 n-gram보다 잘 포착한다.</li>
  <li>Positional Encoding이 없으면 순서 정보가 사라져 문법 패턴 학습이 약해질 수 있다.</li>
  <li>모델이 커질수록 잠재력은 증가하지만, 데이터/학습 안정성 관리가 더 중요해진다.</li>
</ul>

<h3>10. 한계와 다음 실험 계획</h3>
<p><strong>이번 실험에서 배운 점:</strong> {learned}</p>
<p><strong>다음 실험 액션:</strong> {next_actions}</p>
<p>한계: 현재 설정은 학습 step과 모델 크기가 제한적이므로, 논문 수준 성능 재현이라기보다 구조적 유효성 확인에 가깝습니다.</p>
<p>다음 단계: 동일 데이터셋에서 LSTM baseline을 추가하고, seed 3회 평균/분산까지 보고서에 포함하면 신뢰도가 크게 올라갑니다.</p>

<h3>11. 개선 아이디어</h3>
<ul>
  <li>하이퍼파라미터 탐색 자동화</li>
  <li>다중 시드 평균 지표 도입</li>
  <li>실패 케이스 자동 분류</li>
</ul>

<h3>12. 실무 적용 가능성</h3>
<p>연구 로그 자동화 및 지표 추적을 통해 팀 단위 실험 운영에 바로 적용 가능합니다.</p>
{qa_block}

<h3>이 포스팅을 읽고 직접 설명할 수 있어야 하는 핵심 3가지</h3>
<ul>
  <li>이 논문이 해결하려는 문제와 기존 방식의 한계</li>
  <li>핵심 개념(Attention/Positional Encoding)이 코드에서 어떤 역할을 하는지</li>
  <li>실험 그래프를 보고 학습 상태를 어떻게 해석하는지</li>
</ul>

<h3>다음에 스스로 해볼 실습 2가지</h3>
<ul>
  <li>학습률을 바꿔 수렴 속도와 최종 지표 변화를 비교하기</li>
  <li>seed를 여러 개로 바꿔 평균 성능과 분산을 기록하기</li>
</ul>
""".strip()

        return BlogDraft(title=title, tags=tags, html=html)
