# AlignForge · 反向模式自动微分微内核（tape-based reverse-mode AD）
# 站在 PyTorch autograd / JAX autodiff 的经典设计之上：每个 Op 记录前向值与局部梯度闭包，
# 反向遍历 tape 累积梯度。所有 op 手写并经中心差分梯度检验交叉验证。
# 作者：晨星
from __future__ import annotations
import numpy as np


def unbroadcast(g: np.ndarray, shape: tuple) -> np.ndarray:
    """把梯度规约回原形状（处理广播）。规约规则同 torch.autograd。"""
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


class Node:
    __slots__ = ("v", "g", "parents", "bwd", "name")
    _id = 0

    def __init__(self, v, parents=(), bwd=None, name=None):
        self.v = np.asarray(v, dtype=np.float64)
        self.g = None
        self.parents = parents
        self.bwd = bwd
        self.name = name

    def backward(self):
        """从该节点反向传播（该节点梯度置 1）。要求图为 DAG。
        开始前清空整图累积梯度，避免跨图累加（同一参数节点被复用时会残留）。"""
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            n, done = stack.pop()
            if done:
                topo.append(n)
                continue
            if id(n) in seen:
                continue
            seen.add(id(n))
            stack.append((n, True))
            for p in n.parents:
                if id(p) not in seen:
                    stack.append((p, False))
        for n in topo:
            n.g = None
        self.g = np.ones_like(self.v)
        for n in reversed(topo):
            if n.bwd is None or not n.parents:
                continue
            for p, g in zip(n.parents, n.bwd(n.g)):
                if g is None:
                    continue
                p.g = g if p.g is None else p.g + g

    def __getitem__(self, key):
        return slice_node(self, key)

    # ---- 便捷操作符 ----
    def __add__(self, o): return add(self, o if isinstance(o, Node) else const(o))
    __radd__ = __add__

    def __mul__(self, o): return mul(self, o if isinstance(o, Node) else const(o))
    __rmul__ = __mul__

    def __sub__(self, o): return sub(self, o if isinstance(o, Node) else const(o))

    def __neg__(self): return scale(self, -1.0)


def const(v, name=None):
    return Node(v, (), None, name)


def param(v, name=None):
    return Node(v, (), None, name or f"p{Node._id}")


# ---------------- 一元算子 ----------------

def _unary(x: Node, v, deriv):
    def bwd(g):
        return (deriv(g, x.v),)
    return Node(v, (x,), bwd)


def silu(x):
    """SiLU/Swish：x·σ(x)。反向 σ(x)(1+x(1−σ(x)))。"""
    s = 1.0 / (1.0 + np.exp(-x.v))
    return _unary(x, x.v * s, lambda g, v: g * s * (1.0 + v * (1.0 - s)))


def exp(x):
    v = np.exp(x.v)
    return _unary(x, v, lambda g, vv: g * vv)


def rsqrt(x):
    v = 1.0 / np.sqrt(x.v)
    return _unary(x, v, lambda g, vv: g * (-0.5) * vv ** 3)


def neg(x):
    return _unary(x, -x.v, lambda g, vv: -g)


def scale(x, c):
    return _unary(x, x.v * c, lambda g, vv: g * c)


def softmax_rows(x):
    """行 softmax（数值稳定）。反向 s·(g − Σ g·s)。"""
    z = x.v - x.v.max(axis=-1, keepdims=True)
    e = np.exp(z)
    s = e / e.sum(axis=-1, keepdims=True)
    def bwd(g):
        return (s * (g - (g * s).sum(axis=-1, keepdims=True)),)
    return Node(s, (x,), bwd)


def logsumexp_rows(x):
    z = x.v - x.v.max(axis=-1, keepdims=True)
    lse = np.log(np.exp(z).sum(axis=-1, keepdims=True)) + x.v.max(axis=-1, keepdims=True)
    return _unary(x, lse, lambda g, vv: g * np.exp(vv - lse))


def tanh(x):
    v = np.tanh(x.v)
    return _unary(x, v, lambda g, vv: g * (1 - vv * vv))


# ---------------- 二元算子 ----------------

def add(a: Node, b: Node):
    def bwd(g):
        return (unbroadcast(g, a.v.shape), unbroadcast(g, b.v.shape))
    return Node(a.v + b.v, (a, b), bwd)


def sub(a: Node, b: Node):
    def bwd(g):
        return (unbroadcast(g, a.v.shape), unbroadcast(-g, b.v.shape))
    return Node(a.v - b.v, (a, b), bwd)


def mul(a: Node, b: Node):
    def bwd(g):
        return (unbroadcast(g * b.v, a.v.shape), unbroadcast(g * a.v, b.v.shape))
    return Node(a.v * b.v, (a, b), bwd)


def matmul(a: Node, b: Node):
    def bwd(g):
        ga = g @ np.swapaxes(b.v, -1, -2)
        gb = np.swapaxes(a.v, -1, -2) @ g
        if ga.shape != a.v.shape:
            ga = unbroadcast(ga, a.v.shape)
        if gb.shape != b.v.shape:
            gb = unbroadcast(gb, b.v.shape)
        return (ga, gb)
    return Node(a.v @ b.v, (a, b), bwd)


# ---------------- 形算子 / 归约 ----------------

def reshape(x: Node, shape):
    def bwd(g):
        return (g.reshape(x.v.shape),)
    return Node(x.v.reshape(shape), (x,), bwd)


def transpose(x: Node, axes):
    inv = np.argsort(axes)
    def bwd(g):
        return (g.transpose(inv),)
    return Node(x.v.transpose(axes), (x,), bwd)


def slice_node(x: Node, key):
    """切片算子：x[key]，反向把梯度散回原形状。"""
    def bwd(g):
        full = np.zeros_like(x.v)
        full[key] += g
        return (full,)
    return Node(x.v[key], (x,), bwd)


def minimum(a: Node, b: Node):
    v = np.minimum(a.v, b.v)
    def bwd(g):
        return (np.where(a.v <= b.v, g, 0.0), np.where(b.v < a.v, g, 0.0))
    return Node(v, (a, b), bwd)


def maximum(a: Node, b: Node):
    v = np.maximum(a.v, b.v)
    def bwd(g):
        return (np.where(a.v >= b.v, g, 0.0), np.where(b.v > a.v, g, 0.0))
    return Node(v, (a, b), bwd)


def clip(a: Node, lo: float, hi: float):
    v = np.clip(a.v, lo, hi)
    def bwd(g):
        return (np.where((a.v > lo) & (a.v < hi), g, 0.0),)
    return Node(v, (a,), bwd)


def log_sigmoid(x):
    """log σ(x) = −softplus(−x)，数值稳定实现。反向 σ(x)。"""
    v = -np.maximum(-x.v, 0) - np.log1p(np.exp(-np.abs(x.v)))
    s = 1.0 / (1.0 + np.exp(-x.v))
    return _unary(x, v, lambda g, vv: g * s)


def sum_axis(x: Node, axis=-1):
    n = x.v.shape[axis]
    def bwd(g):
        return (np.broadcast_to(np.expand_dims(g, axis), x.v.shape).copy(),)
    return Node(x.v.sum(axis=axis), (x,), bwd)


def sum_all(x: Node):
    return _unary(x, np.array(x.v.sum()), lambda g, vv: g * np.ones_like(x.v))


def mean_axis(x: Node, axis=-1, keepdims=True):
    n = x.v.shape[axis]
    def bwd(g):
        if not keepdims:
            g = np.expand_dims(g, axis)
        return (np.broadcast_to(g / n, x.v.shape).copy(),)
    return Node(x.v.mean(axis=axis, keepdims=keepdims), (x,), bwd)


# ---------------- 融合算子（降低反向拼装出错面） ----------------

def rmsnorm(x: Node, g_node: Node, eps: float = 1e-6):
    """RMSNorm：y = x/rms(x)·g。反向用闭式公式。"""
    xv = x.v
    D = xv.shape[-1]
    ms = (xv * xv).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(ms + eps)
    s = xv * inv
    y = s * g_node.v
    def bwd(g):
        gg = g * g_node.v
        inner = (gg * s).sum(axis=-1, keepdims=True) / D
        dx = gg * inv - s * inner * inv
        dg = (g * s).sum(axis=tuple(range(g.ndim - 1)))
        return (dx, dg)
    return Node(y, (x, g_node), bwd)


def linear(x: Node, W: Node, b: Node | None = None):
    y = matmul(x, W)
    if b is not None:
        y = add(y, b)
    return y
