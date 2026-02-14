from __future__ import annotations
from typing import Union
import math


class Value:
    """The Scalar Object for Neural Networks"""

    Number = Union[float, int, "Value"]

    def __init__(
        self,
        _data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        _label: str = "",
    ) -> None:
        self.data = _data
        self.label = _label
        self._prev: set[Value] = set(_children)
        self._op: str = _op
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other: Number) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other: Number) -> Value:
        return self + other

    def __mul__(self, other: Number) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other: Number) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Number) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return self + (-other)

    def __rsub__(self, other: Number) -> Value:
        return other + (-self)

    def exp(self) -> Value:
        x = self.data
        res = math.exp(x)

        out = Value(res, (self,), "exp")

        def _backward():
            self.grad += res * out.grad

        out._backward = _backward

        return out

    def __pow__(self, n: Union[float, int]) -> Value:
        res = self.data**n

        out = Value(res, (self,), f"**{n}")

        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad

        out._backward = _backward

        return out

    def __truediv__(self, other: Number) -> Value:
        return self * (other**-1)

    def tanh(self) -> Value:
        e = math.exp(2 * self.data)
        t = Value(((e - 1) / (e + 1)), (self,), "tanh")

        def _backward():
            self.grad += (1 - (t.data**2)) * t.grad

        t._backward = _backward

        return t

    def backward(self):
        topo: list[Value] = []
        seen: set[Value] = set()

        def build_topo(root: Value):
            if root not in seen:
                seen.add(root)

                for v in root._prev:
                    build_topo(v)

                topo.append(root)

        build_topo(self)

        self.grad = 1.0
        nodes = reversed(topo)

        for node in nodes:
            node._backward()

        return nodes
