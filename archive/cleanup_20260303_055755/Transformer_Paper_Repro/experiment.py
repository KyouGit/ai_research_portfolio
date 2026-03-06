from __future__ import annotations

import logging
import math
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch import Tensor
from tqdm import tqdm


@dataclass(slots=True)
class Vocab:
    stoi: dict[str, int]
    itos: list[str]
    unk_id: int = 0

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 4096,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x: Tensor) -> Tensor:
        seq_len, batch = x.shape
        pos = torch.arange(seq_len, device=x.device).unsqueeze(1).expand(seq_len, batch)
        h = self.token_emb(x) + self.pos_emb(pos)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.encoder(h, mask=mask)
        h = self.norm(h)
        return self.head(h)


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize_text(lines: list[str]) -> list[str]:
    tokens: list[str] = []
    for line in lines:
        line = line.strip().lower()
        if not line:
            continue
        tokens.extend(line.split())
    return tokens


def build_vocab(tokens: list[str], max_vocab_size: int = 20000) -> Vocab:
    counter = Counter(tokens)
    most_common = [token for token, _ in counter.most_common(max_vocab_size - 2)]
    itos = ["<unk>", "<pad>"] + most_common
    stoi = {token: idx for idx, token in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos, unk_id=0)


def batchify(ids: list[int], batch_size: int, device: torch.device) -> Tensor:
    data = torch.tensor(ids, dtype=torch.long)
    usable = (data.size(0) // batch_size) * batch_size
    data = data[:usable]
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def get_batch(source: Tensor, idx: int, bptt: int) -> tuple[Tensor, Tensor]:
    seq_len = min(bptt, source.size(0) - 1 - idx)
    data = source[idx : idx + seq_len]
    target = source[idx + 1 : idx + 1 + seq_len].reshape(-1)
    return data, target


def evaluate(model: TransformerLM, data: Tensor, bptt: int, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for i in range(0, max(1, data.size(0) - 1), bptt):
            if i >= data.size(0) - 1:
                break
            x, y = get_batch(data, i, bptt)
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y)
            losses.append(float(loss.item()))
    avg_loss = float(np.mean(losses)) if losses else float("inf")
    ppl = float(math.exp(avg_loss)) if avg_loss < 20 else float("inf")
    return avg_loss, ppl


def run_experiment(paper_name: str, paper_dir: Path, seed: int = 42, iterations: int = 300) -> dict:
    set_seed(seed)

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

    logger.info("experiment_start paper=%s seed=%s iterations=%s", paper_name, seed, iterations)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    ds_train = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ds_valid = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")

    train_tokens = tokenize_text(ds_train["text"])
    valid_tokens = tokenize_text(ds_valid["text"])
    vocab = build_vocab(train_tokens, max_vocab_size=20000)

    train_ids = vocab.encode(train_tokens)
    valid_ids = vocab.encode(valid_tokens)

    batch_size = 16
    eval_batch_size = 16
    bptt = 35

    train_data = batchify(train_ids, batch_size=batch_size, device=device)
    valid_data = batchify(valid_ids, batch_size=eval_batch_size, device=device)

    model = TransformerLM(vocab_size=len(vocab.itos)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    max_steps = max(20, iterations)
    stride = bptt
    max_start = max(1, train_data.size(0) - bptt - 1)

    train_losses: list[float] = []
    eval_steps: list[int] = []
    eval_ppls: list[float] = []

    iterator = tqdm(range(1, max_steps + 1), desc=f"{paper_name} train") if max_steps >= 100 else range(1, max_steps + 1)
    for step in iterator:
        model.train()
        start = ((step - 1) * stride) % max_start
        x, y = get_batch(train_data, start, bptt)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_loss = float(loss.item())
        train_losses.append(train_loss)

        if step == 1 or step % 20 == 0 or step == max_steps:
            val_loss, val_ppl = evaluate(model, valid_data, bptt, criterion)
            eval_steps.append(step)
            eval_ppls.append(val_ppl)
            logger.info(
                "step=%s train_loss=%.4f val_loss=%.4f val_ppl=%.4f",
                step,
                train_loss,
                val_loss,
                val_ppl,
            )

    final_ppl = eval_ppls[-1] if eval_ppls else float("inf")
    score = 0.0 if not math.isfinite(final_ppl) else 1.0 / (1.0 + final_ppl)

    fig_path = results_dir / f"metric_curve_{run_stamp}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    if eval_steps:
        plt.plot(eval_steps, eval_ppls, marker="o", label="val_perplexity")
    plt.xlabel("step")
    plt.ylabel("metric")
    plt.title(f"{paper_name} on WikiText-2")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    result_txt = results_dir / f"result_{run_stamp}.txt"
    result_txt.write_text(
        "\n".join(
            [
                f"paper_name: {paper_name}",
                f"run_at_utc: {utc_now_iso()}",
                f"seed: {seed}",
                f"iterations: {max_steps}",
                "dataset: wikitext-2-raw-v1",
                f"vocab_size: {len(vocab.itos)}",
                f"final_val_perplexity: {final_ppl}",
                f"score: {score:.6f}",
                f"figure_path: {fig_path}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    logger.info("experiment_done final_val_ppl=%s score=%.6f", final_ppl, score)
    return {
        "paper_name": paper_name,
        "score": score,
        "note": f"wikitext2_val_ppl={final_ppl}",
        "result_path": str(result_txt),
        "log_path": str(log_path),
        "image_path": str(fig_path),
    }


if __name__ == "__main__":
    base = Path(__file__).resolve().parent
    print(run_experiment("Transformer Paper Repro", base, seed=42, iterations=300))
