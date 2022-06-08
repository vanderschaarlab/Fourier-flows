import numpy as np
import pytest
from fflows import FourierFlow
from data import sine_data_generation


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("FFT", [True, False])
@pytest.mark.parametrize("flip", [True, False])
def test_training(normalize: bool, FFT: bool, flip: bool) -> None:
    T = 11
    n_samples = 100
    dims = 3
    X = sine_data_generation(no=n_samples, seq_len=T, dim=3)

    ff_params = {
        "hidden": 11,
        "n_flows": 11,
        "normalize": normalize,
        "FFT": FFT,
        "flip": flip,
    }
    train_params = {
        "epochs": 10,
        "batch_size": 500,
        "learning_rate": 1e-3,
        "display_step": 100,
    }

    model = FourierFlow(**ff_params)
    _ = model.fit(X, **train_params)


def test_generation() -> None:
    T = 11
    n_samples = 100
    dims = 3
    X = sine_data_generation(no=n_samples, seq_len=T, dim=dims)

    ff_params = {"hidden": 11, "n_flows": 11, "normalize": False}
    train_params = {
        "epochs": 10,
        "batch_size": 500,
        "learning_rate": 1e-3,
        "display_step": 100,
    }

    model = FourierFlow(**ff_params)
    _ = model.fit(X, **train_params)

    samples = model.sample(10)

    assert samples.shape == (10, T, dims)
