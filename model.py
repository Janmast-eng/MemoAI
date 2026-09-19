from __future__ import annotations

import json
import os
import re
import sys
import time

import numpy as np

from gpu import RNG, scatter_add, to_device, to_host, xp

F32 = np.float32
SQRT_2_OVER_PI = F32(np.sqrt(2.0 / np.pi))


def configure_utf8():
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

def softmax(x, axis=-1):
    x = x - xp.max(x, axis=axis, keepdims=True)
    e = xp.exp(x)
    return e / xp.sum(e, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1.0 + xp.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)))


def gelu_grad(x):
    t = xp.tanh(SQRT_2_OVER_PI * (x + 0.044715 * x * x * x))
    return (
        0.5 * (1.0 + t)
        + 0.5 * x * (1.0 - t * t) * SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * x * x)
    )


def _td2(a, b):
    return xp.tensordot(a, b, axes=([0, 1], [0, 1]))


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.g = xp.ones(dim, dtype=F32)
        self.b = xp.zeros(dim, dtype=F32)
        self.eps = F32(eps)

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = xp.mean(xc * xc, axis=-1, keepdims=True)
        std = xp.sqrt(var + self.eps)
        xn = xc / std
        return self.g * xn + self.b, (xn, std)

    def backward(self, dout, cache):
        xn, std = cache
        n = F32(dout.shape[-1])
        dgamma = xp.sum(dout * xn, axis=tuple(range(dout.ndim - 1)))
        dbeta = xp.sum(dout, axis=tuple(range(dout.ndim - 1)))
        dxn = dout * self.g
        s1 = xp.sum(dxn, axis=-1, keepdims=True)
        s2 = xp.sum(dxn * xn, axis=-1, keepdims=True)
        dx = (dxn * n - s1 - xn * s2) / (n * std)
        return dx, dgamma, dbeta

def default_config(vocab_size: int) -> dict:
    return {
        "vocab_size": int(vocab_size),
        "n_layer": 4,
        "n_head": 4,
        "n_embd": 128,
        "block_size": 128,
    }


class NanoGPT:

    def __init__(self, config: dict, seed: int = 1337):
        self.c = dict(config)
        c = self.c
        V, C, L, H = c["vocab_size"], c["n_embd"], c["n_layer"], c["n_head"]
        assert C % H == 0, "n_embd 必须能被 n_head 整除"
        c["head_dim"] = C // H

        rng = RNG(seed)
        self.P = {}

        def w(shape, std):
            return rng.normal(0.0, std, size=shape).astype(F32)

        self.P["tok_emb"] = w((V, C), 0.02)
        self.P["pos_emb"] = w((c["block_size"], C), 0.02)

        res_scale = 1.0 / np.sqrt(2.0 * L)
        for i in range(L):
            self.P[f"attn_q_{i}"] = w((C, C), 0.02)
            self.P[f"attn_k_{i}"] = w((C, C), 0.02)
            self.P[f"attn_v_{i}"] = w((C, C), 0.02)
            self.P[f"attn_o_{i}"] = w((C, C), 0.02 * res_scale)
            self.P[f"ffn_1_{i}"] = w((C, 4 * C), 0.02)
            self.P[f"ffn_2_{i}"] = w((4 * C, C), 0.02 * res_scale)
        self.P["head"] = w((C, V), 0.02)

        self.ln1 = [LayerNorm(C) for _ in range(L)]
        self.ln2 = [LayerNorm(C) for _ in range(L)]
        self.lnf = LayerNorm(C)

        self._masks: dict[int, np.ndarray] = {}
        self.cache = None

    def _norms(self):
        for i in range(self.c["n_layer"]):
            yield f"ln1_{i}", self.ln1[i]
            yield f"ln2_{i}", self.ln2[i]
        yield "lnf", self.lnf

    def state_dict(self) -> dict:
        sd = {k: v for k, v in self.P.items()}
        for name, ln in self._norms():
            sd[name + ".g"] = ln.g
            sd[name + ".b"] = ln.b
        return sd

    def load_state_dict(self, sd: dict):
        for k in self.P:
            self.P[k] = to_device(sd[k])
        for name, ln in self._norms():
            ln.g = to_device(sd[name + ".g"])
            ln.b = to_device(sd[name + ".b"])

    def n_params(self) -> int:
        return int(sum(v.size for v in self.state_dict().values()))

    def _mask(self, T: int):
        m = self._masks.get(T)
        if m is None:
            m = xp.triu(xp.full((T, T), F32(-1e9), dtype=F32), k=1)
            self._masks[T] = m
        return m

    def forward(self, idx, targets=None):
        """
        idx: (B, T) int；targets: (B, T) int 或 None
        返回 (logits, loss)；loss 为 None 时不做 softmax/交叉熵。
        输入可以是主机端数组，进来后自动搬到计算设备。
        """
        c = self.c
        idx = to_device(idx)
        if targets is not None:
            targets = to_device(targets)
        B, T = idx.shape
        C, H, D, L = c["n_embd"], c["n_head"], c["head_dim"], c["n_layer"]
        P = self.P
        scale = F32(1.0 / np.sqrt(D))
        mask = self._mask(T)

        x = P["tok_emb"][idx] + P["pos_emb"][:T]  # (B,T,C)
        layers = []

        for i in range(L):
            h, ln1_cache = self.ln1[i].forward(x)

            q = (h @ P[f"attn_q_{i}"]).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            k = (h @ P[f"attn_k_{i}"]).reshape(B, T, H, D).transpose(0, 2, 1, 3)
            v = (h @ P[f"attn_v_{i}"]).reshape(B, T, H, D).transpose(0, 2, 1, 3)

            scores = (q @ k.transpose(0, 1, 3, 2)) * scale + mask  # (B,H,T,T)
            att = softmax(scores, axis=-1)
            y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, C)  # (B,T,C)
            aout = y @ P[f"attn_o_{i}"]

            x = x + aout

            h2, ln2_cache = self.ln2[i].forward(x)
            u = h2 @ P[f"ffn_1_{i}"]  # (B,T,4C)
            a = gelu(u)
            f = a @ P[f"ffn_2_{i}"]  # (B,T,C)
            x = x + f

            layers.append(
                dict(h=h, ln1=ln1_cache, q=q, k=k, v=v, att=att, y=y, h2=h2, u=u, a=a, ln2=ln2_cache)
            )

        xf, lnf_cache = self.lnf.forward(x)
        logits = xf @ P["head"]  # (B,T,V)

        loss = None
        dlogits = None
        if targets is not None:
            V = c["vocab_size"]
            flat = logits.reshape(-1, V)
            tgt = targets.reshape(-1)
            probs = softmax(flat, axis=-1)
            rows = xp.arange(tgt.size)
            loss = float(-xp.log(probs[rows, tgt] + 1e-8).mean())
            d = probs
            d[rows, tgt] -= 1.0
            d /= F32(tgt.size)
            dlogits = d.reshape(B, T, V)

        self.cache = dict(layers=layers, xf=xf, lnf=lnf_cache, idx=idx, dlogits=dlogits, T=T)
        return logits, loss

    def backward(self, dlogits=None) -> dict:
        c = self.c
        cache = self.cache
        B, T = cache["idx"].shape
        C, H, D, L = c["n_embd"], c["n_head"], c["head_dim"], c["n_layer"]
        P = self.P
        layers = cache["layers"]
        if dlogits is None:
            dlogits = cache["dlogits"]
        assert dlogits is not None, "需要 dlogits：请用 forward(idx, targets) 后再 backward()"

        G = {k: xp.zeros_like(v) for k, v in P.items()}

        xf = cache["xf"]
        G["head"] = _td2(xf, dlogits)                     # (C,V)
        dx = dlogits @ P["head"].T                        # (B,T,C)
        dx, dg, db = self.lnf.backward(dx, cache["lnf"])
        G["lnf.g"], G["lnf.b"] = dg, db

        for i in reversed(range(L)):
            lay = layers[i]

            # ---- FFN: x = x2 + gelu(x3@W1)@W2 ----
            G[f"ffn_2_{i}"] = _td2(lay["a"], dx)           # (4C,C)
            da = dx @ P[f"ffn_2_{i}"].T
            du = da * gelu_grad(lay["u"])
            G[f"ffn_1_{i}"] = _td2(lay["h2"], du)          # (C,4C)
            dh2 = du @ P[f"ffn_1_{i}"].T
            dx2_ffn, dg, db = self.ln2[i].backward(dh2, lay["ln2"])
            G[f"ln2_{i}.g"], G[f"ln2_{i}.b"] = dg, db
            # out = x2 + f：除 FFN 分支外还有一条残差直连分支，梯度要相加
            dx2 = dx + dx2_ffn                              # dL/dx2

            daout = dx2                                     # x2 = x + aout
            G[f"attn_o_{i}"] = _td2(lay["y"], daout)       # (C,C)
            dy = (daout @ P[f"attn_o_{i}"].T).reshape(B, T, H, D).transpose(0, 2, 1, 3)

            att, q, k, v = lay["att"], lay["q"], lay["k"], lay["v"]
            dv = att.transpose(0, 1, 3, 2) @ dy            # (B,H,T,D)
            datt = dy @ v.transpose(0, 1, 3, 2)            # (B,H,T,T)
            ds = att * (datt - xp.sum(datt * att, axis=-1, keepdims=True))
            ds = ds * F32(1.0 / np.sqrt(D))
            dq = ds @ k
            dk = ds.transpose(0, 1, 3, 2) @ q

            dqf = dq.transpose(0, 2, 1, 3).reshape(B, T, C)
            dkf = dk.transpose(0, 2, 1, 3).reshape(B, T, C)
            dvf = dv.transpose(0, 2, 1, 3).reshape(B, T, C)

            G[f"attn_q_{i}"] = _td2(lay["h"], dqf)
            G[f"attn_k_{i}"] = _td2(lay["h"], dkf)
            G[f"attn_v_{i}"] = _td2(lay["h"], dvf)
            dh = dqf @ P[f"attn_q_{i}"].T + dkf @ P[f"attn_k_{i}"].T + dvf @ P[f"attn_v_{i}"].T

            dx_res, dg, db = self.ln1[i].backward(dh, lay["ln1"])
            G[f"ln1_{i}.g"], G[f"ln1_{i}.b"] = dg, db
            dx = dx2 + dx_res

        demb = dx
        dtok = G["tok_emb"]
        scatter_add(dtok, cache["idx"].reshape(-1), demb.reshape(-1, C))
        G["pos_emb"][:T] = demb.sum(axis=0)
        return G

class AdamW:
    def __init__(self, params: dict, lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = {k: xp.zeros_like(v) for k, v in params.items()}
        self.v = {k: xp.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict):
        self.t += 1
        lr_t = self.lr
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, p in params.items():
            g = grads.get(k)
            if g is None:
                continue
            m = self.m[k]
            v = self.v[k]
            m *= self.b1
            m += (1.0 - self.b1) * g
            v *= self.b2
            v += (1.0 - self.b2) * (g * g)
            mhat = m / bc1
            vhat = v / bc2
            p -= lr_t * (mhat / (xp.sqrt(vhat) + self.eps) + self.wd * p)

    def state_dict(self):
        return {"m": self.m, "v": self.v, "t": np.array([self.t])}

    def load_state_dict(self, sd):
        self.m = {k: v for k, v in sd["m"].items()}
        self.v = {k: v for k, v in sd["v"].items()}
        self.t = int(sd["t"][0])


def clip_grad_norm(grads: dict, max_norm: float = 1.0) -> float:
    if not grads:
        return 0.0
    # 全程在计算设备上求平方和，只做一次主机同步（避免逐参数 float() 反复同步）
    total = float(xp.sqrt(xp.sum(xp.stack([xp.sum(g * g) for g in grads.values()]))))
    if total > max_norm and total > 0:
        s = max_norm / total
        for k in grads:
            grads[k] *= s
    return total

class CharTokenizer:
    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
    SPECIALS = (PAD, UNK, BOS, EOS)

    def __init__(self, text: str | None = None, max_vocab: int = 0):
        self.itos: list[str] = [self.PAD, self.UNK, self.BOS, self.EOS]
        if text:
            counts: dict[str, int] = {}
            for ch in text:
                counts[ch] = counts.get(ch, 0) + 1
            ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            # max_vocab <= 0 表示不限制词表大小
            limit = len(ordered) if max_vocab <= 0 else max(0, max_vocab - len(self.itos))
            for ch, _ in ordered[:limit]:
                self.itos.append(ch)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk_id = self.stoi[self.UNK]

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids if int(i) < len(self.itos))

    def token_bytes(self, i) -> bytes:
        i = int(i)
        if i < len(self.SPECIALS) or i >= len(self.itos):
            return b""
        return self.itos[i].encode("utf-8", errors="ignore")

    def is_special(self, i) -> bool:
        return int(i) < len(self.SPECIALS)

    def meta(self) -> dict:
        return {"type": "char"}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tk = cls()
        tk.itos = data["itos"]
        tk.stoi = {s: i for i, s in enumerate(tk.itos)}
        tk.unk_id = tk.stoi[cls.UNK]
        return tk


# ---------------------------------------------------------------------------
# 纯 NumPy 实现的 BPE（Byte Pair Encoding）分词器
#   - 训练：整段语料摊平成一条 int32 数组，pair 统计 / 全局合并全部走 numpy 向量化
#   - 编解码：GPT-2 式 byte-level 词表（每个字节映射到一个可打印码位），
#     任何 UTF-8 文本都能无损还原，不会出现 <unk>
# ---------------------------------------------------------------------------

def _byte_encoder() -> dict[int, int]:
    """字节 -> 可打印码位（GPT-2 的做法），保证词表能安全存进 JSON。"""
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\xa1"), ord("\xac") + 1))
        + list(range(ord("\xae"), ord("\xff") + 1))
    )
    codes = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            codes.append(256 + n)
            n += 1
    return dict(zip(printable, codes))


BYTE_ENCODER = _byte_encoder()
BYTE_DECODER = {code: byte for byte, code in BYTE_ENCODER.items()}
BASE_SYMBOLS = [chr(BYTE_ENCODER[b]) for b in range(256)]

# 预分词：先把文本切成一个个「不能再跨」的小块，BPE 只在块内部合并。
# 规则顺序是有意义的：缩读 -> CJK/日文/韩文单字 -> 字母串 -> 数字串 -> 标点 -> 空白
_TOKEN_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d"
    r"| ?[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]"
    r"| ?[^\W\d_]+"
    r"| ?[0-9]+"
    r"| ?[^\s\w]+"
    r"|\s+"
    r"|[\s\S]"
)


def pretokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text)


class BPETokenizer:
    PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
    SPECIALS = (PAD, UNK, BOS, EOS)
    N_SPECIAL = len(SPECIALS)
    BASE = 256

    # np.bincount 的 key 空间上限（约 200MB int64），超过就退化到 np.unique
    _MAX_BINCOUNT = 25_000_000

    def __init__(self, itos: list[str] | None = None, merges: list[tuple[int, int]] | None = None):
        self.itos = list(self.SPECIALS) + list(BASE_SYMBOLS) if itos is None else list(itos)
        self.ranks: dict[tuple[int, int], int] = {}
        for a, b in merges or []:
            self.ranks[(int(a), int(b))] = len(self.ranks)
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        self.unk_id = self.stoi[self.UNK]
        self._cache: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self.itos)

    @property
    def merges(self) -> list[tuple[int, int]]:
        return [k for k, _ in sorted(self.ranks.items(), key=lambda kv: kv[1])]

    def _new_id(self, rank: int) -> int:
        return self.N_SPECIAL + self.BASE + rank

    # ---------------------------------------------------------------- 训练
    @classmethod
    def train(cls, text: str, vocab_size: int = 2048, min_pair_freq: int = 2,
              max_chars: int | None = None, verbose: bool = True) -> "BPETokenizer":
        if max_chars and len(text) > max_chars:
            text = text[:max_chars]

        target = max(0, int(vocab_size) - cls.N_SPECIAL - cls.BASE)
        width = max(1, int(vocab_size)) + 1          # key = a * width + b
        use_bincount = width * width <= cls._MAX_BINCOUNT

        chunks = pretokenize(text)
        if not chunks:
            raise ValueError("语料为空，无法训练 BPE 分词器")

        # 所有块首尾相接摊平成一条数组，块与块之间插入哨兵 0（<pad>，永远不参与合并）。
        # 这样任何跨块的相邻 pair 都不可能被统计到，后续合并可以整条数组一次做完。
        whole = text.encode("utf-8")
        buf = np.frombuffer(whole, dtype=np.uint8).astype(np.int32)
        SEP = np.zeros(1, dtype=np.int32)
        pieces: list[np.ndarray] = []
        pos = 0
        for ch in chunks:
            n = len(ch.encode("utf-8"))
            pieces.append(buf[pos:pos + n] + cls.N_SPECIAL)
            pieces.append(SEP)
            pos += n
        pieces.append(SEP)
        ids = np.concatenate(pieces)
        del pieces

        itos = list(cls.SPECIALS) + list(BASE_SYMBOLS)
        merges: list[tuple[int, int]] = []
        t0 = time.time()

        while len(merges) < target:
            a, b = ids[:-1], ids[1:]
            ok = (a >= cls.N_SPECIAL) & (b >= cls.N_SPECIAL)
            if not ok.any():
                break
            keys = a[ok].astype(np.int64) * width + b[ok]

            if use_bincount:
                counts = np.bincount(keys, minlength=width * width)
                best = int(np.argmax(counts))
                freq = int(counts[best])
            else:
                uniq, cnts = np.unique(keys, return_counts=True)
                j = int(np.argmax(cnts))
                best, freq = int(uniq[j]), int(cnts[j])

            if freq < max(1, min_pair_freq):
                break

            pa, pb = divmod(best, width)
            new_id = cls.N_SPECIAL + cls.BASE + len(merges)
            merges.append((pa, pb))
            itos.append(itos[pa] + itos[pb])
            ids = _merge_once(ids, np.int32(pa), np.int32(pb), np.int32(new_id))

            if verbose and (len(merges) % 100 == 0 or len(merges) == target):
                el = time.time() - t0
                eta = el / len(merges) * (target - len(merges))
                # ascii() 避免在某些 Windows 控制台编码下打印非 ASCII 直接崩溃
                pair = ascii(_piece_text(itos[pa] + itos[pb], errors="backslashreplace"))
                print(f"  [bpe] {len(merges)}/{target} merge {pair} x{freq} "
                      f"| {el:.0f}s, eta {eta:.0f}s", flush=True)

        if verbose:
            print(f"[bpe] vocab {len(itos):,} (256 base + {len(merges):,} merges), "
                  f"{time.time() - t0:.1f}s", flush=True)
        return cls(itos=itos, merges=merges)

    # ---------------------------------------------------------------- 编码
    def encode(self, text: str) -> list[int]:
        out: list[int] = []
        for ch in pretokenize(text):
            out.extend(self._encode_chunk(ch))
        return out

    def _encode_chunk(self, chunk: str) -> list[int]:
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit
        raw = np.frombuffer(chunk.encode("utf-8"), dtype=np.uint8)
        ids = (raw.astype(np.int32) + self.N_SPECIAL).tolist()
        while len(ids) > 1:
            best_rank, best_i = None, -1
            for i in range(len(ids) - 1):
                r = self.ranks.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_rank is None:
                break
            ids[best_i:best_i + 2] = [self._new_id(best_rank)]
        if len(self._cache) < 500_000:
            self._cache[chunk] = ids
        return ids

    def decode(self, ids) -> str:
        return self.bytes_of(ids).decode("utf-8", errors="replace")

    def bytes_of(self, ids) -> bytes:
        out = bytearray()
        for i in ids:
            out += self.token_bytes(i)
        return bytes(out)

    def token_bytes(self, i) -> bytes:
        i = int(i)
        if i < self.N_SPECIAL or i >= len(self.itos):
            return b""
        return bytes(BYTE_DECODER.get(ord(ch), ord(ch) & 0xFF) for ch in self.itos[i])

    def is_special(self, i) -> bool:
        return int(i) < self.N_SPECIAL

    def piece(self, i) -> str:
        return _piece_text(self.itos[int(i)])

    def meta(self) -> dict:
        return {"type": "bpe", "merges": [[a, b] for a, b in self.merges]}

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "bpe", "itos": self.itos, **self.meta()}, f, ensure_ascii=False)

    @classmethod
    def from_pieces(cls, itos: list[str], merges) -> "BPETokenizer":
        return cls(itos=itos, merges=[(int(a), int(b)) for a, b in merges])

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_pieces(data["itos"], data["merges"])


def piece_bytes(symbol: str) -> bytes:
    """把内部使用的「字节符号串」还原成原始字节。"""
    return bytes(BYTE_DECODER.get(ord(ch), ord(ch) & 0xFF) for ch in symbol)


def _piece_text(symbol: str, errors: str = "replace") -> str:
    return piece_bytes(symbol).decode("utf-8", errors=errors)


def _merge_once(ids: np.ndarray, pa: np.int32, pb: np.int32, new_id: np.int32) -> np.ndarray:
    """在整条 id 序列里把相邻 pair (pa,pb) 全部就地合并：向量化，无 Python 逐位循环。

    采用从左往右的贪心（如 'aaa' 合并 'aa' 得到 ('aa')('a')），
    和训练脚本常见的串行实现语义一致。
    """
    n = ids.size
    if n < 2:
        return ids
    hit = (ids[:-1] == pa) & (ids[1:] == pb)
    if not hit.any():
        return ids

    # 标记每次连续命中段的起点，段内按下标奇偶取，即得到互不重叠的合并位置
    start = hit.copy()
    start[1:] &= ~hit[:-1]
    heads = np.flatnonzero(start)
    if heads.size == 0:
        return ids
    group = np.cumsum(start) - 1
    local = np.arange(n - 1) - heads[np.maximum(group, 0)]
    pos = np.flatnonzero(hit & (local % 2 == 0))
    if pos.size == 0:
        return ids

    out = ids.copy()
    out[pos] = new_id
    keep = np.ones(out.size, dtype=bool)
    keep[pos + 1] = False
    return out[keep]


class Utf8Streamer:
    """按 token 喂字节，只有凑成完整 UTF-8 字符才吐出文本（用于流式打印）。"""

    def __init__(self):
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> str:
        if not chunk:
            return ""
        self._buf += chunk
        try:
            text = self._buf.decode("utf-8")
        except UnicodeDecodeError:
            return ""
        self._buf.clear()
        return text

    def flush(self) -> str:
        text = self._buf.decode("utf-8", errors="replace")
        self._buf.clear()
        return text


def tokenizer_from_meta(meta: dict):
    """根据 meta.json 里的信息还原分词器（兼容旧的字符级 checkpoint）。"""
    info = meta.get("tokenizer") or {}
    itos = meta["itos"]
    if info.get("type") == "bpe" and info.get("merges"):
        return BPETokenizer.from_pieces(itos, info["merges"])
    tk = CharTokenizer()
    tk.itos = list(itos)
    tk.stoi = {s: i for i, s in enumerate(tk.itos)}
    tk.unk_id = tk.stoi[tk.UNK]
    return tk

# ---------------------------------------------------------------------------
# 数据集读取：仅支持 .jsonl（每行一个 JSON 对象）
#   支持的常见对话 / 文本格式：
#     {"conversation": [{"human": "...", "assistant": "..."}, ...]}   # 本仓库使用的格式
#     {"conversations": [{"from": "human", "value": "..."}, ...]}     # ShareGPT
#     {"messages": [{"role": "user", "content": "..."}, ...]}         # OpenAI
#     {"instruction": ..., "input": ..., "output": ...}
#     {"prompt": ..., "completion": ...} / {"question":..., "answer":...}
#     {"text": "..."}，或直接一行纯字符串
# ---------------------------------------------------------------------------

TEXT_EXT = {".jsonl"}

HUMAN_ROLE = "human"
ASSISTANT_ROLE = "assistant"

# 训练和推理共用的对话文本格式：
#   用户: 问题
#   助手: 回答
USER_PREFIX = "用户:"
ASSISTANT_PREFIX = "助手:"
TURN_SEP = "\n"
DIALOG_SEP = "\n\n"

_ROLE_ALIASES = {
    "human": HUMAN_ROLE, "user": HUMAN_ROLE, "persona": HUMAN_ROLE,
    "assistant": ASSISTANT_ROLE, "gpt": ASSISTANT_ROLE, "bot": ASSISTANT_ROLE,
    "ai": ASSISTANT_ROLE, "model": ASSISTANT_ROLE, "chatgpt": ASSISTANT_ROLE,
}
_SEQ_KEYS = ("conversation", "conversations", "messages", "dialog", "dialogue", "turns", "qa", "data")
_ROLE_KEYS = ("from", "role", "speaker", "who")
_TEXT_KEYS = ("value", "content", "text", "message")
_HUMAN_KEYS = ("instruction", "prompt", "question", "query", "input")
_ASSISTANT_KEYS = ("output", "response", "answer", "completion", "target", "reply")


def _as_text(v) -> str:
    if isinstance(v, str):
        return v.strip()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return str(v)
    return ""


def _first_text(obj: dict, keys) -> str:
    for k in keys:
        t = _as_text(obj.get(k))
        if t:
            return t
    return ""


def _turns_from_seq(items) -> list[tuple[str, str]]:
    """把一轮对话列表（human/assistant 交替）解析成 [(role, text), ...]。"""
    turns: list[tuple[str, str]] = []
    if not isinstance(items, list):
        return turns
    for it in items:
        if isinstance(it, str):
            turns.append((ASSISTANT_ROLE, it.strip()))
            continue
        if not isinstance(it, dict):
            continue

        role = None
        for k in _ROLE_KEYS:
            raw = it.get(k)
            if isinstance(raw, str):
                role = _ROLE_ALIASES.get(raw.strip().lower())
                if role:
                    break

        if role is None:
            # 老的 {"human": ..., "assistant": ...} 双键格式
            h, a = _as_text(it.get("human")), _as_text(it.get("assistant"))
            if h:
                turns.append((HUMAN_ROLE, h))
            if a:
                turns.append((ASSISTANT_ROLE, a))
            continue

        text = _first_text(it, _TEXT_KEYS)
        if text:
            turns.append((role, text))
    return [(r, t) for r, t in turns if t]


def _dialog_from_obj(obj) -> list[tuple[str, str]]:
    """把 .jsonl 的一行解析成一段对话 [(role, text), ...]，解析不出内容则返回空列表。"""
    if isinstance(obj, str):
        t = obj.strip()
        return [(ASSISTANT_ROLE, t)] if t else []
    if not isinstance(obj, dict):
        return []

    for key in _SEQ_KEYS:
        seq = obj.get(key)
        if isinstance(seq, list):
            turns = _turns_from_seq(seq)
            if turns:
                return turns

    question = _first_text(obj, _HUMAN_KEYS)
    if question and _as_text(obj.get("instruction")):
        extra = _as_text(obj.get("input"))
        if extra and extra != question:
            question = question + "\n" + extra
    answer = _first_text(obj, _ASSISTANT_KEYS)
    if question and answer:
        return [(HUMAN_ROLE, question), (ASSISTANT_ROLE, answer)]

    text = _first_text(obj, ("text", "content"))
    if text:
        return [(ASSISTANT_ROLE, text)]
    return []


def dialog_to_text(turns) -> str:
    """一段对话 -> 训练用的纯文本（与推理时的 prompt 格式保持一致）。"""
    return TURN_SEP.join(
        (USER_PREFIX if role == HUMAN_ROLE else ASSISTANT_PREFIX) + " " + text
        for role, text in turns
    )


def dialogs_to_text(dialogs) -> str:
    return DIALOG_SEP.join(dialog_to_text(d) for d in dialogs)


def encode_dialogs(tok, dialogs, add_bos: bool = True, add_eos: bool = True) -> list[int]:
    """把对话逐段编码成一条 token 流：每段以 BOS 开头、EOS 结尾。"""
    bos = tok.stoi.get(getattr(tok, "BOS", "<bos>"))
    eos = tok.stoi.get(getattr(tok, "EOS", "<eos>"))
    ids: list[int] = []
    for turns in dialogs:
        if add_bos and bos is not None:
            ids.append(int(bos))
        ids.extend(tok.encode(dialog_to_text(turns)))
        if add_eos and eos is not None:
            ids.append(int(eos))
    return ids


def read_jsonl(path: str) -> tuple[list[list[tuple[str, str]]], int]:
    """读取单个 .jsonl 文件，返回 (对话列表, 无法解析的行数)。"""
    dialogs: list[list[tuple[str, str]]] = []
    bad = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                bad += 1
                continue
            turns = _dialog_from_obj(obj)
            if turns:
                dialogs.append(turns)
            else:
                bad += 1
    return dialogs, bad


def load_corpus(data_path: str) -> tuple[list[list[tuple[str, str]]], list[str], int]:
    """加载数据集（.jsonl 文件或包含 .jsonl 的目录）。

    返回 (对话列表, 文件列表, 跳过行数)；用 dialogs_to_text() 可得到拼接后的纯文本。
    """
    files: list[str] = []
    if os.path.isdir(data_path):
        for root, dirs, names in os.walk(data_path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for n in sorted(names):
                if os.path.splitext(n)[1].lower() in TEXT_EXT:
                    files.append(os.path.join(root, n))
        files.sort()
    elif os.path.isfile(data_path):
        if os.path.splitext(data_path)[1].lower() not in TEXT_EXT:
            raise FileNotFoundError(f"只支持 .jsonl 数据集：{data_path}")
        files.append(data_path)

    if not files:
        raise FileNotFoundError(
            f"在 {data_path} 下没有找到任何 .jsonl 文件（本程序只读取 jsonl 数据集）"
        )

    dialogs: list[list[tuple[str, str]]] = []
    bad = 0
    for p in files:
        d, b = read_jsonl(p)
        dialogs.extend(d)
        bad += b
    if not dialogs:
        raise FileNotFoundError(f"{data_path} 中没有解析出任何有效样本，请检查 jsonl 内容")
    return dialogs, files, bad


def save_checkpoint(model_dir: str, name: str, model: NanoGPT, opt: AdamW, meta: dict):
    os.makedirs(model_dir, exist_ok=True)
    sd = model.state_dict()
    payload = {f"p:{k}": to_host(v) for k, v in sd.items()}
    payload.update({f"m:{k}": to_host(v) for k, v in opt.m.items()})
    payload.update({f"v:{k}": to_host(v) for k, v in opt.v.items()})
    payload["opt_t"] = np.array([opt.t], dtype=np.int64)
    np.savez(os.path.join(model_dir, name), **payload)

    m = dict(meta)
    m["config"] = model.c
    with open(os.path.join(model_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    stem = os.path.splitext(name)[0]
    if stem != "model":
        with open(os.path.join(model_dir, stem + ".json"), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=2)
    return os.path.join(model_dir, name)


def _newest_ckpt(model_dir: str) -> str | None:
    if not os.path.isdir(model_dir):
        return None
    cands = [f for f in os.listdir(model_dir) if re.match(r"ckpt_epoch_\d+\.npz$", f)]
    if cands:
        return os.path.join(model_dir, sorted(cands)[-1])
    if os.path.isfile(os.path.join(model_dir, "model.npz")):
        return os.path.join(model_dir, "model.npz")
    return None


def load_checkpoint(model_dir: str, ckpt: str | None = None):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"模型目录不存在: {model_dir}（请先运行 train.py）")
    meta_path = os.path.join(model_dir, "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"缺少 {meta_path}，模型目录可能不完整")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    path = ckpt or _newest_ckpt(model_dir)
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(f"在 {model_dir} 下没有找到可用的 checkpoint")

    side = os.path.join(model_dir, os.path.splitext(os.path.basename(path))[0] + ".json")
    if os.path.isfile(side):
        with open(side, "r", encoding="utf-8") as f:
            meta = {**meta, **json.load(f)}

    data = np.load(path)
    model = NanoGPT(meta["config"])
    sd = {k[2:]: to_device(data[k]) for k in data.files if k.startswith("p:")}
    model.load_state_dict(sd)

    opt = AdamW(model.state_dict())
    opt.m = {k[2:]: to_device(data[k]) for k in data.files if k.startswith("m:")}
    opt.v = {k[2:]: to_device(data[k]) for k in data.files if k.startswith("v:")}
    opt.t = int(data["opt_t"][0]) if "opt_t" in data.files else 0

    tk = tokenizer_from_meta(meta)
    return model, tk, meta, path, opt

def _softmax_host(x):
    """主机端 softmax（sample_token 取回 logits 后使用，避免误用 GPU 后端）。"""
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def sample_token(logits, rng, temperature=1.0, top_k: int = 0, top_p: float = 1.0) -> int:
    # 采样在主机端完成：先取回 logits（GPU 模式下仅回传一个 vocab 大小的向量）
    logits = to_host(logits).astype(np.float64).reshape(-1)
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k and 0 < top_k < logits.size:
        idx = np.argpartition(logits, -top_k)[-top_k:]
        mask = np.full(logits.size, -np.inf)
        mask[idx] = logits[idx]
        logits = mask
    if 0.0 < top_p < 1.0:
        order = np.argsort(-logits)
        probs = _softmax_host(logits[order])
        cum = np.cumsum(probs)
        cut = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:cut]
        mask = np.full(logits.size, -np.inf)
        mask[keep] = logits[keep]
        logits = mask
    probs = _softmax_host(logits)
    probs = probs / probs.sum()
    return int(rng.choice(probs.size, p=probs))
