"""计算后端选择：优先 CuPy（NVIDIA CUDA），否则回退 NumPy（CPU）。

- 自动检测：安装了 CuPy 且存在可用 NVIDIA GPU 时启用 GPU
- 强制 CPU：设置环境变量 WAFIGPT_DEVICE=cpu，或命令行加 --cpu
- 布局：模型参数与前向/反向全部在计算设备上执行；
  分词、数据读取、批次采样保持在主机端，仅把 batch 搬上 GPU。
"""
from __future__ import annotations

import os
import sys

import numpy as np

USE_GPU = False
_CUPY_IMPORTED = False

def _force_cpu() -> bool:
    v = os.environ.get("WAFIGPT_DEVICE", "").strip().lower()
    if v in ("cpu", "-1"):
        return True
    if v in ("cuda", "gpu", "nvidia"):
        return False
    return "--cpu" in sys.argv


if not _force_cpu():
    try:
        import warnings
        with warnings.catch_warnings():
            # 使用 pip 安装的 nvidia-* 运行库时该警告必然出现且无害
            warnings.filterwarnings("ignore", message="CUDA path could not be detected")
            import cupy as cp
        import cupyx

        _CUPY_IMPORTED = True
        if cp.cuda.is_available():
            cp.cuda.Device(0).use()
            USE_GPU = True
    except Exception:
        USE_GPU = False

# 计算后端：numpy 或 cupy（模块对象，后端在导入期固定，全程不再切换）
xp = cp if USE_GPU else np


def to_device(a):
    """主机数组 -> 计算设备（GPU 模式返回 CuPy 数组，CPU 模式原样返回）。"""
    if a is None or isinstance(a, xp.ndarray):
        return a
    return xp.asarray(a)


def to_host(a):
    """计算设备 -> 主机端（npz 保存、Python 数值运算时使用）。"""
    if _CUPY_IMPORTED and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def scatter_add(arr, indices, values):
    """arr[indices] += values：NumPy 用 add.at，CuPy 用 cupyx.scatter_add。"""
    if USE_GPU:
        cupyx.scatter_add(arr, indices, values)
    else:
        np.add.at(arr, indices, values)


class RNG:
    """统一随机数接口。

    - 模型初始化（normal）在计算设备上生成
    - 批次采样（integers）与 token 采样（choice）固定在主机端
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._host = np.random.default_rng(seed)
        if USE_GPU:
            cp.random.seed(seed)

    def normal(self, loc=0.0, scale=1.0, size=None):
        if USE_GPU:
            return cp.random.normal(loc, scale, size=size)
        return self._host.normal(loc, scale, size)

    def integers(self, low, high=None, size=None):
        return self._host.integers(low, high, size=size)

    def choice(self, n, p=None):
        if p is not None:
            p = to_host(p)
        return int(self._host.choice(n, p=p))


def describe_device() -> str:
    """返回当前计算后端的描述信息（用于启动日志）。"""
    if USE_GPU:
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        free, total = cp.cuda.runtime.memGetInfo()
        return (f"GPU: {name} | 显存 {total / 2**30:.1f} GB（空闲 {free / 2**30:.1f} GB）"
                f" | CuPy {cp.__version__}")
    msg = f"CPU（NumPy {np.__version__}）"
    if not _CUPY_IMPORTED:
        msg += " | 提示：安装 CuPy 可启用 N 卡加速（CUDA 12.x: pip install cupy-cuda12x）"
    else:
        msg += " | 已装 CuPy 但未检测到可用的 NVIDIA GPU/CUDA 驱动"
    return msg
