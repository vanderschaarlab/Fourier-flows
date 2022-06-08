from fflows import FourierFlow
import numpy as np


def sine_data_generation(no, seq_len, dim, freq_scale=1):

    """Sine data generation.

    Args:

    - no: the number of samples
    - seq_len: sequence length of the time-series
    - dim: feature dimensions

    Returns:
    - data: generated data

    """
    # Initialize the output

    data = list()

    # Generate sine data

    for i in range(no):

        # Initialize each time-series
        temp = list()

        # For each feature
        for k in range(dim):

            # Randomly drawn frequency and phase
            # freq      = np.random.uniform(0, 0.1)
            # phase     = np.random.uniform(0, 0.1)

            freq = np.random.beta(2, 2)  # np.random.uniform(0, 0.1)
            phase = np.random.normal()

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq_scale * freq * j + phase) for j in range(seq_len)]
            # temp_data = [np.exp(-1 * freq * j) * np.sin(5 * freq * j + phase) for j in range(seq_len)]
            # temp_data = [np.sinc(freq * j + phase) for j in range(seq_len)]

            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))

        # Stack the generated data
        data.append(temp)

    return np.asarray(data)

def test_sanity() -> None:
    T           = 101
    n_samples   = 1000
    X           = sine_data_generation(no=n_samples, seq_len=T, dim=1)

    ff_params = {"hidden": 11, "fft_size" : T, "n_flows"  : 10, "normalize"  : False}
    train_params =  {
                "epochs": 100,
                "batch_size": 500,
                "learning_rate": 1e-3,
                "display_step": 100,
            }

    data = sine_data_generation(100, seq_len = T, dim = 7)
    print(data.shape)

    model = FourierFlow(**ff_params)
    _ = model.fit(data, **train_params)
