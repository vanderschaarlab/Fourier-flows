import numpy as np
import pytest
from fflows import FourierFlow
from data import sine_data_generation


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("FFT", [True, False])
@pytest.mark.parametrize("flip", [True, False])
@pytest.mark.parametrize("T", [12, 11])
@pytest.mark.parametrize("dims", [4, 6, 3])
@pytest.mark.parametrize("n_flows", [5, 10, 15])
def test_training(
    normalize: bool, FFT: bool, flip: bool, T: int, dims: int, n_flows: int
) -> None:
    n_samples = 100
    X = sine_data_generation(no=n_samples, seq_len=T, dim=dims)

    ff_params = {
        "hidden": 11,
        "n_flows": n_flows,
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


@pytest.mark.parametrize("T", [12, 11])
@pytest.mark.parametrize("dims", [4, 3])
def test_generation(T: int, dims: int) -> None:
    n_samples = 100
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
