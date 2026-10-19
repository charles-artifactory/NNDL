"""Five optimizers implemented directly with NumPy arrays."""
import numpy as np


class Optimizer:
    def __init__(self, name, learning_rate, momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8):
        if name not in ("sgd", "momentum", "adagrad", "adam", "adamax"):
            raise ValueError(f"Unknown optimizer: {name}")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.name, self.lr = name, learning_rate
        self.momentum, self.beta1, self.beta2, self.epsilon = momentum, beta1, beta2, epsilon
        self.step_count = 0
        self.first, self.second = {}, {}

    def step(self, params, grads):
        self.step_count += 1
        for key, p in params.items():
            g = grads[key]
            if not np.isfinite(g).all():
                raise FloatingPointError(f"Non-finite gradient: {key}")
            if self.name == "sgd":
                update = g
            elif self.name == "momentum":
                velocity = self.first.setdefault(key, np.zeros_like(p))
                velocity *= self.momentum
                velocity += g
                update = velocity
            elif self.name == "adagrad":
                squares = self.second.setdefault(key, np.zeros_like(p))
                squares += g * g
                update = g / (np.sqrt(squares) + self.epsilon)
            else:
                first = self.first.setdefault(key, np.zeros_like(p))
                second = self.second.setdefault(key, np.zeros_like(p))
                first *= self.beta1
                first += (1 - self.beta1) * g
                if self.name == "adam":
                    second *= self.beta2
                    second += (1 - self.beta2) * g * g
                    vhat = second / (1 - self.beta2 ** self.step_count)
                    denominator = np.sqrt(vhat) + self.epsilon
                else:
                    np.maximum(self.beta2 * second, np.abs(g), out=second)
                    denominator = second + self.epsilon
                mhat = first / (1 - self.beta1 ** self.step_count)
                update = mhat / denominator
            p -= self.lr * update
