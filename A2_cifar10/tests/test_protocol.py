import numpy as np
import pytest
import torch
from torch import nn

from cifar10_lab.data import batch_transform, occlude
from cifar10_lab.experiment import epoch_pass, final_specs, loader, metrics, seed_all
from cifar10_lab.models import Classifier, ModelSpec, normalization


@pytest.mark.parametrize("norm", ["none", "batch", "instance", "layer", "group"])
def test_normalizations_train_and_predict(norm):
    model = Classifier(ModelSpec(normalization=norm, width=8))
    x = torch.randn(3, 3, 32, 32)
    logits = model(x)
    assert logits.shape == (3, 10)
    nn.functional.cross_entropy(logits, torch.tensor([0, 4, 9])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_layer_normalization_matches_all_feature_elements():
    x = torch.randn(2, 8, 16, 16)
    norm = normalization("layer", 8, 16)
    assert isinstance(norm, nn.LayerNorm)
    y = norm(x)
    assert torch.allclose(y.mean((1, 2, 3)), torch.zeros(2), atol=1e-6)
    assert norm.weight.shape == (8, 16, 16)


@pytest.mark.parametrize("kernel,depth,residual", [(3, 1, False), (5, 1, False), (7, 1, False), (3, 2, True)])
def test_architectures_preserve_classification_shape(kernel, depth, residual):
    model = Classifier(ModelSpec(kernel_size=kernel, depth=depth, residual=residual, width=8))
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 10)


def test_validation_preprocessing_is_deterministic_and_does_not_mutate():
    x = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
    original = x.clone()
    y = batch_transform(x)
    assert torch.equal(y, batch_transform(x))
    assert torch.equal(original, x)
    assert y.min() >= -1 and y.max() <= 1
    cut = occlude(y, 8)
    assert torch.count_nonzero(cut[:, :, 12:20, 12:20]) == 0
    assert torch.equal(y[:, :, :12], cut[:, :, :12])


def test_training_augmentation_is_seeded_and_preserves_shape():
    x = torch.randint(0, 256, (8, 3, 32, 32), dtype=torch.uint8)
    seed_all(42)
    first = batch_transform(x, True, True)
    seed_all(42)
    second = batch_transform(x, True, True)
    assert first.shape == x.shape
    assert torch.equal(first, second)
    assert all((im == 0).sum() >= 3 * 64 for im in first)


def test_loss_is_sample_weighted_with_short_last_batch():
    class Fixed(nn.Module):
        def forward(self, x):
            # First channel stores class signal after normalization.
            a = x[:, 0, 0, 0]
            return torch.stack((a, -a), 1)
    x = torch.zeros((3, 3, 32, 32), dtype=torch.uint8)
    x[2] = 255
    y = torch.tensor([0, 0, 0])
    actual = epoch_pass(Fixed(), loader(x, y, np.arange(3), 2), torch.device("cpu"))
    expected = nn.functional.cross_entropy(Fixed()(batch_transform(x)), y).item()
    assert actual["loss"] == pytest.approx(expected, abs=1e-6)


def test_metrics_keep_missing_classes_and_count_samples():
    result = metrics(np.eye(10)[[0, 1, 1]], np.array([0, 1, 2]))
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert np.array(result["confusion_matrix"]).sum() == 3
    assert len(result["per_class"]) == 10


@pytest.mark.parametrize("selected_dropout", [0.0, 0.2, 0.5])
def test_final_controls_include_positive_and_zero_dropout(selected_dropout):
    specs = final_specs(ModelSpec(dropout=selected_dropout))
    selected, _ = specs["selected"]
    variant, _ = specs["dropout_variant"]
    assert selected.normalization == variant.normalization
    assert sorted([selected.dropout, variant.dropout])[0] == 0
    assert max(selected.dropout, variant.dropout) > 0
    no_norm, _ = specs["no_normalization"]
    assert no_norm.normalization == "none"
    assert no_norm.dropout == max(selected.dropout, variant.dropout)


@pytest.fixture(scope="session", autouse=True)
def few_threads():
    torch.set_num_threads(2)
