import ast
from pathlib import Path

import numpy as np
import pytest

from a1_numpy.data import stratified_split
from a1_numpy.metrics import classification_metrics, evaluate
from a1_numpy.network import MLP, loss_and_gradient, softmax
from a1_numpy.optim import Optimizer


@pytest.mark.parametrize("loss,smoothing,l2,dropout", [
    ("cross_entropy", 0.0, 0.0, 0.0),
    ("mse", 0.0, 0.0, 0.0),
    ("cross_entropy", 0.1, 0.0, 0.0),
    ("cross_entropy", 0.0, 0.03, 0.0),
    ("cross_entropy", 0.0, 0.0, 0.25),
    ("mse", 0.1, 0.03, 0.25),
])
def test_network_finite_difference(loss, smoothing, l2, dropout):
    rng = np.random.default_rng(17)
    x, y = rng.normal(size=(5, 4)), np.array([0, 1, 2, 1, 0])
    model = MLP([4, 5, 3], seed=9, dropout=dropout, dtype=np.float64)

    def objective():
        logits = model.forward(x, training=True, rng=np.random.default_rng(91))
        value, _ = loss_and_gradient(logits, y, loss, smoothing)
        return value + model.penalty(l2)

    logits = model.forward(x, training=True, rng=np.random.default_rng(91))
    _, derivative = loss_and_gradient(logits, y, loss, smoothing)
    analytic = model.backward(derivative, l2)
    epsilon = 1e-6
    for name, parameter in model.params.items():
        numerical = np.empty_like(parameter)
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + epsilon
            upper = objective()
            parameter[index] = original - epsilon
            lower = objective()
            parameter[index] = original
            numerical[index] = (upper - lower) / (2 * epsilon)
        np.testing.assert_allclose(analytic[name], numerical, rtol=2e-5, atol=2e-8)


def test_softmax_stability_and_shift_invariance():
    logits = np.array([[10000.0, 9999.0, -10000.0], [-10000.0, 0.0, 10000.0]])
    p = softmax(logits)
    np.testing.assert_allclose(p.sum(axis=1), 1)
    np.testing.assert_allclose(p, softmax(logits - 35000))
    value, gradient = loss_and_gradient(logits, np.array([2, 0]))
    assert np.isfinite(value) and np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient.sum(axis=1), 0, atol=1e-15)


@pytest.mark.parametrize("name,displacement", [
    ("sgd", 0.2), ("momentum", 0.29),
    ("adagrad", 0.1 + 0.1 / np.sqrt(2)), ("adam", 0.2), ("adamax", 0.2),
])
def test_optimizer_two_steps(name, displacement):
    params = {"w": np.array([1.0, -1.0]), "b": np.array([2.0])}
    grads = {"w": np.array([1.0, -1.0]), "b": np.array([1.0])}
    opt = Optimizer(name, 0.1)
    opt.step(params, grads)
    opt.step(params, grads)
    np.testing.assert_allclose(params["w"], [1 - displacement, -1 + displacement], atol=1e-8)
    np.testing.assert_allclose(params["b"], [2 - displacement], atol=1e-8)
    assert opt.step_count == 2


def test_metrics_and_missing_predictions():
    m = classification_metrics(np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, 1, 1]), 3)
    assert m["accuracy"] == 3 / 5
    assert m["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 1, 0]]
    np.testing.assert_allclose(m["precision"], [1, 0.5, 0])
    np.testing.assert_allclose(m["recall"], [0.5, 1, 0])
    np.testing.assert_allclose(m["f1"], [2/3, 2/3, 0])


def test_evaluation_weights_partial_batch():
    rng = np.random.default_rng(1)
    x, y = rng.normal(size=(7, 4)), np.array([0, 1, 2, 1, 0, 2, 0])
    model = MLP([4, 5, 3], dtype=np.float64)
    full = evaluate(model, x, y, batch_size=7, detailed=True)
    partial = evaluate(model, x, y, batch_size=3, detailed=True)
    assert partial["accuracy"] == full["accuracy"]
    assert partial["data_loss"] == pytest.approx(full["data_loss"])
    assert partial["confusion_matrix"] == full["confusion_matrix"]


def test_stratification_is_disjoint_complete_and_reproducible():
    labels = np.repeat(np.arange(3), [17, 22, 31])
    train, val = stratified_split(labels, 19, 7)
    train2, val2 = stratified_split(labels, 19, 7)
    assert len(val) == 19 and len(train) == 51
    assert not set(train).intersection(val)
    np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(70))
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(val, val2)
    assert np.max(np.abs(np.bincount(labels[val]) - np.array([17, 22, 31]) * 19/70)) < 1


def test_save_load_dropout_inference_and_batch_one(tmp_path):
    model = MLP([4, 5, 3], dropout=0.5)
    x = np.ones((1, 4), dtype=np.float32)
    expected = model.forward(x)
    np.testing.assert_array_equal(model.forward(x), expected)
    model.save(tmp_path / "model.npz", {"epoch": 3})
    restored, metadata = MLP.load(tmp_path / "model.npz")
    np.testing.assert_array_equal(restored.forward(x), expected)
    assert metadata == {"epoch": 3}
    assert evaluate(model, x, np.array([0]), detailed=True)["support"] == [1, 0, 0]


def test_training_learns_small_problem():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(48, 4))
    y = (x[:, 0] + x[:, 1] > 0).astype(int)
    model = MLP([4, 12, 2], seed=5, dtype=np.float64)
    optimizer = Optimizer("adam", 0.03)
    before = evaluate(model, x, y)["data_loss"]
    for _ in range(100):
        logits = model.forward(x, training=True)
        _, gradient = loss_and_gradient(logits, y)
        optimizer.step(model.params, model.backward(gradient))
    after = evaluate(model, x, y)
    assert after["data_loss"] < before * 0.1
    assert after["accuracy"] >= 0.98


def test_core_has_no_ml_framework_imports():
    source = Path(__file__).resolve().parents[1] / "src/a1_numpy"
    forbidden = {"torch", "torchvision", "tensorflow", "keras", "jax", "sklearn", "autograd"}
    for file in source.glob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            if isinstance(node, ast.Import):
                assert not forbidden.intersection(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in forbidden
