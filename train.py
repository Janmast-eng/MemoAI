
from __future__ import annotations

import argparse
import math
import os
import time

import numpy as np

from gpu import RNG, USE_GPU, describe_device
from model import (
    AdamW,
    ASSISTANT_PREFIX,
    BPETokenizer,
    CharTokenizer,
    HUMAN_ROLE,
    NanoGPT,
    TURN_SEP,
    USER_PREFIX,
    clip_grad_norm,
    configure_utf8,
    default_config,
    dialogs_to_text,
    encode_dialogs,
    load_checkpoint,
    load_corpus,
    sample_token,
    save_checkpoint,
)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data")


def get_batch(data: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator):
    ix = rng.integers(0, len(data) - block_size - 1, size=(batch_size,))
    off = ix[:, None] + np.arange(block_size)[None, :]
    return data[off], data[off + 1]


def estimate_loss(model: NanoGPT, splits: dict, block_size: int, batch_size: int,
                  iters: int, rng: np.random.Generator) -> dict:
    out = {}
    for name, data in splits.items():
        if len(data) <= block_size + 1:
            out[name] = float("nan")
            continue
        tot = 0.0
        for _ in range(iters):
            x, y = get_batch(data, block_size, batch_size, rng)
            _, loss = model.forward(x, targets=y)
            tot += loss
        out[name] = tot / iters
    return out


def generate_sample(model: NanoGPT, tok, prompt: str, rng: np.random.Generator,
                    max_new: int = 120, temperature: float = 0.8, top_p: float = 0.95) -> str:
    block = model.c["block_size"]
    ids = tok.encode(prompt) or [tok.unk_id]
    for _ in range(max_new):
        ctx = np.array([ids[-block:]], dtype=np.int64)
        logits, _ = model.forward(ctx)
        nxt = sample_token(logits[0, -1], rng, temperature=temperature, top_p=top_p)
        ids.append(nxt)
        if tok.itos[nxt] == tok.EOS:
            break
    return tok.decode(ids)


def print_progress(epoch: int, n_epochs: int, step: int, total: int,
                   loss: float, val_loss: float | None, lr: float, gnorm: float,
                   elapsed: float, global_step: int, total_steps: int,
                   width: int = 20) -> None:
    """单行原地刷新的训练进度条（纯字符实现，无第三方依赖）。"""
    filled = int(width * step / max(1, total))
    bar = "█" * filled + "-" * (width - filled)
    speed = elapsed / max(1, global_step)
    eta = speed * max(0, total_steps - global_step)
    val_s = f" | val {val_loss:.4f}" if val_loss is not None else ""
    msg = (f"epoch {epoch}/{n_epochs} |{bar}| {step}/{total}"
           f" | loss {loss:.4f}{val_s} | lr {lr:.2e} | g {gnorm:.2f}"
           f" | {speed:.2f}s/it | 剩余 {eta / 60:.1f} 分钟")
    print("\r" + msg.ljust(110), end="", flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="纯 NumPy 训练的 nano 自回归语言模型")
    p.add_argument("--data", default=DEFAULT_DATA,
                   help="数据集：单个 .jsonl 文件，或包含 .jsonl 的目录（只读取 jsonl 对话数据）")
    p.add_argument("--model-dir", default=os.path.join(HERE, "model"), help="模型保存目录")
    # 训练
    p.add_argument("--epochs", type=int, default=10, help="训练轮数，每轮结束保存一次")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--block-size", type=int, default=128, help="上下文长度")
    p.add_argument("--steps-per-epoch", type=int, default=0, help="每轮 step 数，0=自动（遍历一遍训练集）")
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr-ratio", type=float, default=0.1, help="余弦退火终点学习率 = lr * ratio")
    p.add_argument("--warmup", type=int, default=100, help="warmup step 数")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100, help="每 N step 打印一次验证 loss")
    p.add_argument("--eval-iters", type=int, default=5)
    # 模型
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=128)
    # 分词
    p.add_argument("--tokenizer", choices=("bpe", "char"), default="bpe",
                   help="分词器：bpe=NumPy 实现的子词分词器（默认），char=字符级（旧版行为）")
    p.add_argument("--vocab-size", type=int, default=2048,
                   help="BPE 目标词表大小（含 4 个特殊 token）")
    p.add_argument("--bpe-chars", type=int, default=0,
                   help="训练 BPE 时使用的语料字符数上限，0=全部（默认）")
    p.add_argument("--min-pair-freq", type=int, default=2,
                   help="BPE：pair 出现次数低于该值就停止合并")
    p.add_argument("--max-vocab", type=int, default=0, help="char 分词器的词表上限，0=不限制（默认）")
    # 数据
    p.add_argument("--val-ratio", type=float, default=0.1)
    # 其它
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true", help="从 model/ 最新 checkpoint 继续训练")
    # 设备
    p.add_argument("--gpu", action="store_true", help="要求使用 NVIDIA GPU（需安装 CuPy，默认自动检测）")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p

def main():
    configure_utf8()
    args = build_parser().parse_args()
    if args.gpu and not USE_GPU:
        raise SystemExit("[error] --gpu 未检测到可用 NVIDIA GPU："
                         "请安装 CuPy（CUDA 12.x: pip install cupy-cuda12x；CUDA 11.x: pip install cupy-cuda11x）")
    print(f"[device] {describe_device()}")
    rng = RNG(args.seed)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"[data] 读取 jsonl 数据集: {args.data}")
    try:
        dialogs, files, bad_lines = load_corpus(args.data)
    except FileNotFoundError as e:
        raise SystemExit(f"[error] {e}\n[error] 请把 .jsonl 对话数据放到 {args.data} 后重试")
    n_turns = sum(len(d) for d in dialogs)
    corpus = dialogs_to_text(dialogs)
    print(f"[data] 文件 {len(files)} 个 | 对话 {len(dialogs):,} 段 | 轮次 {n_turns:,} "
          f"| 字符 {len(corpus):,}" + (f" | 跳过无效行 {bad_lines:,}" if bad_lines else ""))

    model = tok = opt = None
    start_epoch, global_step = 1, 0
    if args.resume:
        try:
            model, tok, meta_prev, _, opt = load_checkpoint(args.model_dir)
            start_epoch = int(meta_prev.get("epoch", 0)) + 1
            global_step = int(meta_prev.get("step", 0))
            print(f"[resume] 从 epoch {start_epoch - 1} 继续，已训练 {global_step} steps")
        except FileNotFoundError as e:
            print(f"[resume] 无可恢复的 checkpoint（{e}），改为从头训练")

    if model is None:
        if args.tokenizer == "bpe":
            sample_n = args.bpe_chars if args.bpe_chars > 0 else None
            print(f"[token] 训练 BPE 分词器（目标词表 {args.vocab_size}，"
                  f"语料采样 {sample_n or len(corpus):,} 字符）")
            tok = BPETokenizer.train(corpus, vocab_size=args.vocab_size,
                                     min_pair_freq=args.min_pair_freq,
                                     max_chars=sample_n)
        else:
            tok = CharTokenizer(corpus, max_vocab=args.max_vocab)
            print(f"[token] 字符级分词器，词表 {len(tok.itos):,}")
        probe = corpus[:5000]
        if tok.decode(tok.encode(probe)) != probe:
            print("[warn] encode/decode 未能无损还原文段，请检查分词器实现")
        cfg = default_config(len(tok.itos))
        cfg.update(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                   block_size=args.block_size)
        model = NanoGPT(cfg, seed=args.seed)
        opt = AdamW(model.state_dict(), lr=args.lr, weight_decay=args.weight_decay)

    ids = np.array(encode_dialogs(tok, dialogs), dtype=np.int32)
    if len(ids) < 4 * (args.block_size + 2):
        print(f"[warn] 语料仅 {len(ids):,} token，建议准备更大的数据集")
    block = min(args.block_size, max(8, (len(ids) - 2) // 4))
    if block != args.block_size:
        model.c["block_size"] = block
        print(f"[data] 语料偏小，上下文长度调整为 {block}")

    n_val = max(block + 2, int(len(ids) * args.val_ratio))
    n_val = min(n_val, len(ids) - block - 2)
    n_val = max(n_val, 1)
    train_ids, val_ids = ids[: len(ids) - n_val], ids[len(ids) - n_val:]

    ratio = len(corpus) / max(1, len(ids))
    print(f"[token] {type(tok).__name__} 词表 {len(tok.itos):,} | 压缩率 {ratio:.2f} 字符/token "
          f"| 训练 {len(train_ids):,} token | 验证 {len(val_ids):,} token")
    print(f"[model] layer={model.c['n_layer']} head={model.c['n_head']} embd={model.c['n_embd']} "
          f"ctx={model.c['block_size']} | 参数量 {model.n_params():,}")

    if len(train_ids) <= block + 1:
        raise SystemExit(f"[error] 训练 token 太少（{len(train_ids)}），无法组成 batch，请增加数据")

    auto_steps = max(1, len(train_ids) // (args.batch_size * block))
    steps_per_epoch = args.steps_per_epoch if args.steps_per_epoch > 0 else auto_steps
    n_epochs = max(1, args.epochs - start_epoch + 1)
    total_steps = max(1, steps_per_epoch * n_epochs)
    lr_min = args.lr * args.min_lr_ratio
    warmup = max(1, min(args.warmup, max(1, total_steps // 10)))

    def lr_at(step: int) -> float:
        if step < warmup:
            return args.lr * (step + 1) / warmup
        prog = min(1.0, max(0.0, (step - warmup) / max(1, total_steps - warmup)))
        return lr_min + 0.5 * (args.lr - lr_min) * (1.0 + math.cos(math.pi * prog))

    def sample_prompt() -> str:
        """取一条验证集对话的首个用户提问，拼成和训练一致的 prompt 用于抽样展示。"""
        for turns in dialogs[-20:]:
            for role, text in turns:
                if role == HUMAN_ROLE:
                    return f"{USER_PREFIX} {text[:80]}{TURN_SEP}{ASSISTANT_PREFIX}"
        return f"{USER_PREFIX} 你好{TURN_SEP}{ASSISTANT_PREFIX}"

    splits = {"train": train_ids, "val": val_ids}
    print(f"[train] 每 epoch {steps_per_epoch} steps，共 {n_epochs} 个 epoch，"
          f"batch={args.batch_size} x {block}")
    print("-" * 60)

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        ep_loss, ep_n, smooth = 0.0, 0, None
        last_val = None
        for s in range(steps_per_epoch):
            opt.lr = lr_at(global_step)
            x, y = get_batch(train_ids, block, args.batch_size, rng)
            _, loss = model.forward(x, targets=y)
            grads = model.backward()
            gnorm = clip_grad_norm(grads, args.grad_clip)
            opt.step(model.state_dict(), grads)

            ep_loss += loss
            ep_n += 1
            smooth = loss if smooth is None else 0.98 * smooth + 0.02 * loss
            global_step += 1

            if args.eval_every and (s + 1) % args.eval_every == 0:
                last_val = estimate_loss(model, {"val": val_ids}, block, args.batch_size,
                                         args.eval_iters, rng)["val"]

            print_progress(epoch, n_epochs, s + 1, steps_per_epoch, smooth, last_val,
                           opt.lr, gnorm, time.time() - t0, global_step, total_steps)

        print()
        train_loss = ep_loss / max(1, ep_n)
        val_loss = estimate_loss(model, splits, block, args.batch_size, args.eval_iters, rng)["val"]
        print(f"[epoch {epoch}] train {train_loss:.4f} | val {val_loss:.4f} "
              f"| 用时 {time.time() - t0:.0f}s")

        try:
            tok.save(os.path.join(args.model_dir, "tokenizer.json"))
        except Exception as e:
            print(f"[warn] 分词器保存失败: {e}")

        meta = {
            "itos": tok.itos,
            "tokenizer": tok.meta(),
            "epoch": epoch,
            "step": global_step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "steps_per_epoch": steps_per_epoch,
            "n_chars": len(corpus),
            "n_dialogs": len(dialogs),
            "n_turns": n_turns,
            "data_format": "jsonl-chat",
            "source_files": len(files),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path = save_checkpoint(args.model_dir, f"ckpt_epoch_{epoch:04d}.npz", model, opt, meta)
        save_checkpoint(args.model_dir, "model.npz", model, opt, meta)
        print(f"[save] {os.path.basename(path)}（同时更新 model.npz）")

        print("[sample] " + generate_sample(model, tok, sample_prompt(), rng, max_new=100)
              .replace("\n", " ")[:200])
        print("-" * 60)

    print(f"[done] 完成，用时 {(time.time() - t0) / 60:.1f} 分钟；模型目录: {args.model_dir}")
    print(f"[done] 对话: python chat.py --model-dir \"{args.model_dir}\"")


if __name__ == "__main__":
    main()
