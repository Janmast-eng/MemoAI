from __future__ import annotations

import argparse
import os

import numpy as np

from gpu import USE_GPU, describe_device
from model import (ASSISTANT_PREFIX, NanoGPT, TURN_SEP, USER_PREFIX, Utf8Streamer,
                   configure_utf8, load_checkpoint, sample_token)

HERE = os.path.dirname(os.path.abspath(__file__))

HELP_TEXT = (
    "/help 显示本帮助\n"
    "/reset 清空上下文\n"
    "/temp 0.8 采样温度\n"
    "/topk 40 top-k 采样，0 关闭\n"
    "/topp 0.95 nucleus 采样\n"
    "/len 200 单次生成最大 token 数\n"
    "/quit 退出"
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="nano 语言模型对话")
    p.add_argument("--model-dir", default=os.path.join(HERE, "model"), help="模型目录")
    p.add_argument("--ckpt", default=None, help="指定 checkpoint 文件，默认取最新")
    p.add_argument("--temp", type=float, default=0.8, help="采样温度")
    p.add_argument("--top-k", type=int, default=0, help="top-k 采样，0=关闭")
    p.add_argument("--top-p", type=float, default=0.95, help="nucleus 采样")
    p.add_argument("--max-new-tokens", type=int, default=200, help="单次生成最大 token 数")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--prompt", default=None, help="非交互模式：只生成一次后退出")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p


def generate(model: NanoGPT, tok, ids: list[int], max_new: int, rng,
             temperature: float, top_k: int, top_p: float, on_token=None) -> list[int]:
    """on_token(idx) 回调收到的是 token id（BPE 下一个 token 可能只是半个 UTF-8 字符）。"""
    block = model.c["block_size"]
    out = list(ids)
    for _ in range(max_new):
        ctx = np.array([out[-block:]], dtype=np.int64)
        logits, _ = model.forward(ctx)
        nxt = sample_token(logits[0, -1], rng, temperature=temperature, top_k=top_k, top_p=top_p)
        out.append(nxt)
        if on_token:
            on_token(nxt)
        if tok.itos[nxt] == tok.EOS:
            break
    return out


def main():
    configure_utf8()
    args = build_parser().parse_args()
    print(f"[device] {describe_device()}")
    model, tok, meta, ckpt_path, _ = load_checkpoint(args.model_dir, args.ckpt)
    rng = np.random.default_rng(args.seed)

    print(f"[chat] checkpoint: {ckpt_path}")
    print(f"[chat] epoch {meta.get('epoch')} | val_loss {meta.get('val_loss')} "
          f"| 分词器 {type(tok).__name__} 词表 {len(tok.itos)} | 上下文 {model.c['block_size']}")
    print("[chat] 输入 /help 查看命令，/quit 退出\n")

    temp, top_k, top_p, max_new = args.temp, args.top_k, args.top_p, args.max_new_tokens
    history: list[int] = []

    def once(prompt: str):
        nonlocal history
        # 与训练数据保持一致：首轮前面加 BOS，之后接着上一轮往下续写
        turn = f"{USER_PREFIX} {prompt}{TURN_SEP}{ASSISTANT_PREFIX}"
        ids = tok.encode(turn)
        if not history:
            ids = [tok.stoi[tok.BOS]] + ids
        else:
            ids = tok.encode(TURN_SEP) + ids
        history = history + ids
        print("model> ", end="", flush=True)
        stream = Utf8Streamer()

        def emit(idx):
            if tok.is_special(idx):
                return
            text = stream.feed(tok.token_bytes(idx))
            if text:
                print(text, end="", flush=True)

        history = generate(model, tok, history, max_new, rng, temp, top_k, top_p, emit)
        tail = stream.flush()
        if tail:
            print(tail, end="", flush=True)
        print("\n")
        # 训练时每段对话以 EOS 结尾，模型一旦吐出 EOS 就表示这段对话结束
        if history and tok.itos[history[-1]] == tok.EOS:
            history = []
        elif len(history) > model.c["block_size"] * 4:
            history = history[-model.c["block_size"]:]

    if args.prompt is not None:
        once(args.prompt)
        return

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[chat] 再见")
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            print("[chat] 再见")
            break
        if user == "/help":
            print(HELP_TEXT)
            continue
        if user == "/reset":
            history = []
            print("[chat] 上下文已清空")
            continue
        if user.startswith("/temp "):
            temp = float(user.split()[1]); print(f"[chat] temp = {temp}"); continue
        if user.startswith("/topk "):
            top_k = int(user.split()[1]); print(f"[chat] top_k = {top_k}"); continue
        if user.startswith("/topp "):
            top_p = float(user.split()[1]); print(f"[chat] top_p = {top_p}"); continue
        if user.startswith("/len "):
            max_new = int(user.split()[1]); print(f"[chat] max_new_tokens = {max_new}"); continue
        once(user)


if __name__ == "__main__":
    main()
