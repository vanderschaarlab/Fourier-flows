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
            freq = np.random.beta(2, 2)  # np.random.uniform(0, 0.1)
            phase = np.random.normal()

            # Generate sine signal based on the drawn frequency and phase
            temp_data = [np.sin(freq_scale * freq * j + phase) for j in range(seq_len)]

            temp.append(temp_data)

        # Align row/column
        temp = np.transpose(np.asarray(temp))

        # Stack the generated data
        data.append(temp)

    return np.asarray(data)
