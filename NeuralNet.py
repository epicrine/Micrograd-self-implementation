from Value import Value
import random


class Module:
    @staticmethod
    def zero_grad(parameters: list[Value]):
        for p in parameters:
            p.grad = 0.0

class Neuron:
    def __init__(self, nin: int):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x: list[Value]):
        act = sum([xi * wi for xi, wi in zip(x, self.w)], self.b)
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: list[Value]):
        outs = [neuron(x) for neuron in self.neurons]
        return outs

    def parameters(self):
        params = [p for neuron in self.neurons for p in neuron.parameters()]
        return params


class MLP:
    def __init__(self, nin: int, nouts: list[int]):
        sz = nouts + [nin]

        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]

    def __call__(self, xs: list[Value]):
        for layer in self.layers:
            xs = layer(xs)
        return xs

    def parameters(self):
        params = [p for layer in self.layers for p in layer.parameters()]
        return params
