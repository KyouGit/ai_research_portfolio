from __future__ import annotations

import json
import logging
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class Doc:
    doc_id: str
    title: str
    text: str


@dataclass(slots=True)
class QAItem:
    qid: str
    question: str
    answer: str
    evidence_doc_id: str
    required_keywords: list[str]


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def tokenize(text: str) -> list[str]:
    cleaned = text.lower()
    for ch in ",.;:()[]{}!?\"'":
        cleaned = cleaned.replace(ch, " ")
    return [t for t in cleaned.split() if t]


def build_corpus() -> list[Doc]:
    return [
        Doc(
            doc_id="d1",
            title="Self-Attention",
            text=(
                "Self-attention lets each token look at every other token in the same sequence. "
                "This improves long-range dependency modeling and removes strict recurrence."
            ),
        ),
        Doc(
            doc_id="d2",
            title="Scaled Dot-Product Attention",
            text=(
                "Attention uses softmax((QK^T)/sqrt(d_k))V. The sqrt(d_k) scaling stabilizes gradients "
                "when key/query dimensions are large."
            ),
        ),
        Doc(
            doc_id="d3",
            title="Multi-Head Attention",
            text=(
                "Multi-head attention projects inputs into multiple subspaces and attends in parallel. "
                "Different heads capture different relations such as syntax and semantics."
            ),
        ),
        Doc(
            doc_id="d4",
            title="Positional Encoding",
            text=(
                "Transformers need positional encoding because attention alone is permutation-invariant. "
                "Sinusoidal encoding injects order information without recurrence."
            ),
        ),
        Doc(
            doc_id="d5",
            title="Encoder-Decoder Structure",
            text=(
                "The encoder builds contextual representations of source tokens. "
                "The decoder uses masked self-attention and cross-attention to generate target tokens autoregressively."
            ),
        ),
        Doc(
            doc_id="d6",
            title="Computational Tradeoff",
            text=(
                "Self-attention is highly parallelizable but has quadratic complexity in sequence length. "
                "This motivates efficient attention variants for long contexts."
            ),
        ),
    ]


def build_qa_set() -> list[QAItem]:
    return [
        QAItem(
            qid="q1",
            question="Why is positional encoding needed in a Transformer?",
            answer="Because attention alone cannot represent token order.",
            evidence_doc_id="d4",
            required_keywords=["order", "permutation"],
        ),
        QAItem(
            qid="q2",
            question="What is the role of sqrt(d_k) in attention?",
            answer="It scales dot products to keep softmax stable and gradients well-behaved.",
            evidence_doc_id="d2",
            required_keywords=["scale", "stable", "gradient"],
        ),
        QAItem(
            qid="q3",
            question="How does multi-head attention help compared to a single head?",
            answer="It attends in multiple subspaces so different relations can be captured in parallel.",
            evidence_doc_id="d3",
            required_keywords=["multiple", "parallel", "relations"],
        ),
        QAItem(
            qid="q4",
            question="What does masked self-attention do in the decoder?",
            answer="It prevents looking at future tokens during autoregressive generation.",
            evidence_doc_id="d5",
            required_keywords=["future", "autoregressive"],
        ),
        QAItem(
            qid="q5",
            question="What key limitation of vanilla self-attention appears on long sequences?",
            answer="Its computation and memory scale quadratically with sequence length.",
            evidence_doc_id="d6",
            required_keywords=["quadratic", "length"],
        ),
        QAItem(
            qid="q6",
            question="Why did Transformer improve over strict recurrent models for context handling?",
            answer="Each token can directly attend to all others, improving long-range dependencies.",
            evidence_doc_id="d1",
            required_keywords=["long-range", "all"],
        ),
    ]


def answer_to_keywords(answer: str, k: int = 3) -> list[str]:
    out: list[str] = []
    for tok in tokenize(answer):
        if len(tok) <= 2 or tok in STOPWORDS:
            continue
        if tok not in out:
            out.append(tok)
        if len(out) >= k:
            break
    return out if out else tokenize(answer)[:1]


def load_squad_subset(
    seed: int,
    cache_dir: Path,
    max_docs: int = 300,
    max_qa: int = 250,
) -> tuple[list[Doc], list[QAItem], str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    ds = load_dataset("squad", split="validation", cache_dir=str(cache_dir.resolve()))
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)

    doc_id_map: dict[str, str] = {}
    docs: list[Doc] = []
    qa_set: list[QAItem] = []

    for i in idxs:
        row = ds[int(i)]
        context = str(row["context"]).strip().replace("\n", " ")
        question = str(row["question"]).strip()
        answers = row.get("answers", {}).get("text", []) if isinstance(row.get("answers"), dict) else []
        answer = str(answers[0]).strip() if answers else ""
        if not context or not question or not answer:
            continue
        if len(context) < 80:
            continue

        if context not in doc_id_map:
            if len(docs) >= max_docs:
                continue
            doc_id = f"d{len(docs) + 1}"
            doc_id_map[context] = doc_id
            title = str(row.get("title", "squad_doc")).strip() or "squad_doc"
            docs.append(Doc(doc_id=doc_id, title=title, text=context))
        evidence_doc_id = doc_id_map[context]

        qa_set.append(
            QAItem(
                qid=f"squad_{row['id']}",
                question=question,
                answer=answer,
                evidence_doc_id=evidence_doc_id,
                required_keywords=answer_to_keywords(answer, k=3),
            )
        )
        if len(qa_set) >= max_qa:
            break

    if len(docs) < 20 or len(qa_set) < 50:
        raise RuntimeError(f"Loaded subset too small: docs={len(docs)} qa={len(qa_set)}")
    return docs, qa_set, f"squad_validation_subset_docs{len(docs)}_qa{len(qa_set)}"


def score_keywords(pred: str, required_keywords: list[str]) -> float:
    pred_tokens = set(tokenize(pred))
    if not required_keywords:
        return 1.0
    hit = 0
    for kw in required_keywords:
        kw_tokens = tokenize(kw)
        matched = True
        for k in kw_tokens:
            if k in pred_tokens:
                continue
            if any(tok.startswith(k) or k.startswith(tok) for tok in pred_tokens):
                continue
            matched = False
            break
        if matched:
            hit += 1
    return hit / len(required_keywords)


def query_expand(query: str) -> str:
    expansions = {
        "order": "position positional permutation",
        "stable": "stability gradient saturation",
        "multi-head": "multiple heads parallel subspaces",
        "future": "autoregressive next-token masking",
        "long": "long-range dependency",
        "complexity": "quadratic memory compute",
    }
    out = [query]
    q = query.lower()
    for key, val in expansions.items():
        if key in q:
            out.append(val)
    return " ".join(out)


def retrieve_topk(
    vectorizer: TfidfVectorizer,
    doc_matrix: np.ndarray,
    query: str,
    docs: list[Doc],
    topk: int = 3,
    expand: bool = False,
) -> list[tuple[Doc, float]]:
    q = query_expand(query) if expand else query
    qv = vectorizer.transform([q])
    sims = cosine_similarity(qv, doc_matrix).ravel()
    order = np.argsort(-sims)[:topk]
    return [(docs[i], float(sims[i])) for i in order]


def build_dense_index(doc_matrix: np.ndarray) -> tuple[TruncatedSVD, np.ndarray]:
    n_components = int(min(64, max(2, doc_matrix.shape[0] - 1), max(2, doc_matrix.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    dense_doc = svd.fit_transform(doc_matrix)
    norms = np.linalg.norm(dense_doc, axis=1, keepdims=True) + 1e-12
    dense_doc = dense_doc / norms
    return svd, dense_doc


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    vecs = embeddings.astype("float32", copy=True)
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def retrieve_topk_dense(
    vectorizer: TfidfVectorizer,
    svd: TruncatedSVD,
    dense_doc_matrix: np.ndarray,
    query: str,
    docs: list[Doc],
    topk: int = 3,
    expand: bool = True,
) -> list[tuple[Doc, float]]:
    q = query_expand(query) if expand else query
    qv_sparse = vectorizer.transform([q])
    qv_dense = svd.transform(qv_sparse)
    qv_dense = qv_dense / (np.linalg.norm(qv_dense, axis=1, keepdims=True) + 1e-12)
    sims = (qv_dense @ dense_doc_matrix.T).ravel()
    order = np.argsort(-sims)[:topk]
    return [(docs[i], float(sims[i])) for i in order]


def retrieve_topk_faiss(
    query: str,
    docs: list[Doc],
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    topk: int = 3,
) -> list[tuple[Doc, float]]:
    q = query_expand(query)
    q_emb = model.encode([q], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    scores, indices = index.search(q_emb, topk)
    out: list[tuple[Doc, float]] = []
    for rank in range(indices.shape[1]):
        di = int(indices[0, rank])
        if di < 0 or di >= len(docs):
            continue
        out.append((docs[di], float(scores[0, rank])))
    return out


def answer_from_doc(doc: Doc, question: str) -> str:
    tokens = set(tokenize(question))
    sents = [s.strip() for s in doc.text.split(".") if s.strip()]
    scored: list[tuple[float, str]] = []
    for s in sents:
        st = set(tokenize(s))
        overlap = len(tokens & st)
        scored.append((overlap, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else doc.text


def run_eval(
    docs: list[Doc],
    qa_set: list[QAItem],
    retrieve_fn,
) -> dict:
    retrieval_correct = 0
    keyword_scores: list[float] = []
    detail_rows: list[dict[str, str]] = []

    for item in qa_set:
        topk = retrieve_fn(item.question, docs, 3)
        pred_doc = topk[0][0]
        pred_answer = answer_from_doc(pred_doc, item.question)
        kscore = score_keywords(pred_answer, item.required_keywords)
        keyword_scores.append(kscore)
        retrieval_hit = pred_doc.doc_id == item.evidence_doc_id
        if retrieval_hit:
            retrieval_correct += 1

        rank_str = " | ".join([f"{i+1}:{d.doc_id}({s:.4f})" for i, (d, s) in enumerate(topk)])
        detail_rows.append(
            {
                "qid": item.qid,
                "question": item.question,
                "pred_doc": pred_doc.doc_id,
                "gold_doc": item.evidence_doc_id,
                "retrieval_hit": str(retrieval_hit),
                "pred_answer": pred_answer,
                "gold_answer": item.answer,
                "topk": rank_str,
                "keyword_score": f"{kscore:.4f}",
            }
        )

    n = len(qa_set)
    retrieval_acc = retrieval_correct / n if n else 0.0
    answer_keyword_f1 = float(np.mean(keyword_scores)) if keyword_scores else 0.0
    score = 0.6 * retrieval_acc + 0.4 * answer_keyword_f1
    return {
        "retrieval_acc": retrieval_acc,
        "answer_keyword_f1": answer_keyword_f1,
        "score": score,
        "rows": detail_rows,
    }


def run_experiment(paper_name: str, paper_dir: Path, seed: int = 42, iterations: int = 0) -> dict:
    del iterations

    results_dir = paper_dir / "results"
    logs_dir = paper_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"run_{run_stamp}.log"

    logger = logging.getLogger(f"experiment.{paper_name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)

    random.seed(seed)
    np.random.seed(seed)

    dataset_name = "internal_transformer_concept_corpus"
    try:
        docs, qa_set, dataset_name = load_squad_subset(
            seed=seed,
            cache_dir=paper_dir / ".hf_cache",
            max_docs=300,
            max_qa=250,
        )
    except Exception:
        docs = build_corpus()
        qa_set = build_qa_set()
    doc_texts = [f"{d.title}. {d.text}" for d in docs]
    min_df = 2 if len(docs) >= 60 else 1
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_features=50000)
    doc_matrix = vectorizer.fit_transform(doc_texts)

    logger.info("task=rag_qa dataset=%s docs=%s qa=%s", dataset_name, len(docs), len(qa_set))

    def sparse_retrieve(query: str, all_docs: list[Doc], topk: int) -> list[tuple[Doc, float]]:
        return retrieve_topk(vectorizer, doc_matrix, query, all_docs, topk=topk, expand=False)

    baseline = run_eval(docs, qa_set, sparse_retrieve)

    retriever_name = "sentence-transformers/all-MiniLM-L6-v2 + FAISS(IndexFlatIP)"
    try:
        emb_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=str((paper_dir / ".hf_cache").resolve()),
            local_files_only=True,
        )
        doc_emb = emb_model.encode(doc_texts, normalize_embeddings=True, convert_to_numpy=True)
        faiss_index = build_faiss_index(doc_emb)

        def dense_retrieve(query: str, all_docs: list[Doc], topk: int) -> list[tuple[Doc, float]]:
            return retrieve_topk_faiss(
                query=query,
                docs=all_docs,
                model=emb_model,
                index=faiss_index,
                topk=topk,
            )

        improved = run_eval(docs, qa_set, dense_retrieve)
    except Exception:
        svd, dense_doc_matrix = build_dense_index(doc_matrix)
        retriever_name = "TF-IDF + LSA dense fallback"

        def dense_retrieve(query: str, all_docs: list[Doc], topk: int) -> list[tuple[Doc, float]]:
            return retrieve_topk_dense(
                vectorizer=vectorizer,
                svd=svd,
                dense_doc_matrix=dense_doc_matrix,
                query=query,
                docs=all_docs,
                topk=topk,
                expand=True,
            )

        improved = run_eval(docs, qa_set, dense_retrieve)

    fig_path = results_dir / f"metric_curve_{run_stamp}.png"
    plt.figure(figsize=(10, 6))
    labels = ["retrieval_acc", "answer_keyword_f1", "overall_score"]
    base_vals = [baseline["retrieval_acc"], baseline["answer_keyword_f1"], baseline["score"]]
    imp_vals = [improved["retrieval_acc"], improved["answer_keyword_f1"], improved["score"]]
    x = np.arange(len(labels))
    width = 0.34
    plt.bar(x - width / 2, base_vals, width=width, label="sparse_tfidf")
    plt.bar(x + width / 2, imp_vals, width=width, label="dense_embedding_lsa")
    plt.xticks(x, labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("score")
    plt.title("RAG QA Evaluation (Higher is Better)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for i, v in enumerate(base_vals):
        plt.text(i - width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    for i, v in enumerate(imp_vals):
        plt.text(i + width / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    improved_rows = improved["rows"]
    improved_better = improved["score"] > baseline["score"] + 1e-9
    score_gain = improved["score"] - baseline["score"]

    if improved_better:
        learned = "Dense 검색 임베딩(LSA) 기반 retriever가 sparse TF-IDF 대비 질의-문서 매칭 품질을 높였다."
    elif abs(score_gain) <= 1e-9:
        learned = "현재 소규모 코퍼스에서는 sparse와 dense 성능이 유사했다. 더 어려운 질의/더 큰 코퍼스로 차이를 키워야 한다."
    else:
        learned = "현재 설정에서는 dense 임베딩 이점이 제한적이어서 임베딩 모델/코퍼스 확장이 필요하다."
    next_actions = "다음 단계로 cross-encoder reranker를 추가해 top-k 재정렬 품질을 측정한다."

    result_txt = results_dir / f"result_{run_stamp}.txt"
    lines = [
        f"paper_name: {paper_name}",
        f"run_at_utc: {utc_now_iso()}",
        f"dataset: {dataset_name}",
        "task_type: rag_qa",
        f"model: TF-IDF baseline vs {retriever_name} + Sentence_Extractor",
        f"doc_count: {len(docs)}",
        f"qa_count: {len(qa_set)}",
        f"baseline_retrieval_acc: {baseline['retrieval_acc']:.6f}",
        f"baseline_answer_keyword_f1: {baseline['answer_keyword_f1']:.6f}",
        f"baseline_score: {baseline['score']:.6f}",
        f"transformer_retrieval_acc: {improved['retrieval_acc']:.6f}",
        f"transformer_answer_keyword_f1: {improved['answer_keyword_f1']:.6f}",
        f"transformer_score: {improved['score']:.6f}",
        f"score_gain_vs_baseline: {score_gain:.6f}",
        f"target_achieved: {str(improved_better)}",
        f"score: {improved['score']:.6f}",
        "metric_definition: score=0.6*retrieval_acc+0.4*answer_keyword_f1",
        f"learned_lessons: {learned}",
        f"next_actions: {next_actions}",
        f"figure_path: {fig_path}",
    ]

    for i, row in enumerate(improved_rows[:3], start=1):
        lines.append(f"example_{i}_type: rag_qa")
        lines.append(f"example_{i}_input: {row['question']}")
        lines.append(f"example_{i}_target: {row['gold_answer']}")
        lines.append(f"example_{i}_topk: {row['topk']}")
        lines.append(f"example_{i}_output: {row['pred_answer']}")
        lines.append(f"example_{i}_evidence_hit: {row['retrieval_hit']}")

    result_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")

    detail_json = results_dir / f"result_{run_stamp}.json"
    detail_json.write_text(
        json.dumps(
            {
                "baseline": baseline,
                "transformer_style": improved,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "done baseline_score=%.4f transformer_score=%.4f gain=%.4f",
        baseline["score"],
        improved["score"],
        score_gain,
    )
    return {
        "paper_name": paper_name,
        "score": improved["score"],
        "note": f"rag_qa_score={improved['score']:.4f},gain={score_gain:.4f}",
        "result_path": str(result_txt),
        "log_path": str(log_path),
        "image_path": str(fig_path),
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    print(run_experiment("Attention Is All You Need", base))
