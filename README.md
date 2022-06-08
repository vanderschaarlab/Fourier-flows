# Fourier-flows


### Description
Code for the [Generative Time-series Modeling with Fourier Flows](https://openreview.net/forum?id=PpshD0AXfA) paper.

Generating synthetic time-series data is crucial in various application domains, such as medical prognosis, wherein research is hamstrung by the lack of access to data due to concerns over privacy. Most of the recently proposed methods for generating synthetic time-series rely on implicit likelihood modeling using generative adversarial networks (GANs)—but such models can be difficult to train, and may jeopardize privacy by “memorizing” temporal patterns in training data. In this paper, we propose an explicit likelihood model based on a novel class of normalizing flows that view time-series data in the frequency-domain rather than the time-domain. The proposed flow, dubbed a Fourier flow, uses a discrete Fourier transform (DFT) to convert variable-length time-series with arbitrary sampling periods into fixed-length spectral representations, then applies a (data-dependent) spectral filter to the frequency-transformed time-series. We show that, by virtue of the DFT analytic properties, the Jacobian determinants and inverse mapping for the Fourier flow can be computed efficiently in linearithmic time, without imposing explicit structural constraints as in existing flows such as NICE (Dinh et al. (2014)), RealNVP (Dinh et al. (2016)) and GLOW (Kingma & Dhariwal (2018)). Experiments show that Fourier flows perform competitively compared to state-of-the-art baselines.

### Installation
* Install with `pip`:
  ```bash
  pip install fflows
  ```

### Examples
Fit a Fourier flow
```python
import numpy as np
from fflows import FourierFlow

T = 11
n_samples = 100
dims = 3
X = np.random.randn(n_samples, T, dims)

model_params = {
    "hidden": 11,
    "n_flows": 11,
}
train_params = {
    "epochs": 10,
    "batch_size": 500,
    "learning_rate": 1e-3,
    "display_step": 100,
}

model = FourierFlow(**model_params)
model.fit(X, **train_params)
```
Generate new data
```python
samples = model.sample(10)

assert samples.shape == (10, T, dims)
```

### Data
The public datasets used in the paper are available at:
* https://drive.google.com/drive/folders/1UILaMFnZpRUf_IhOIkxK2wzBjWBTB86G

### Experiments
* For Experiment 1 (Section 5.1), run [`ICLR 2021 - Experiment 1.ipynb`](./ICLR%202021%20-%20Experiment%201.ipynb).
* For Experiment 2 (Section 5.2), run [`run_experiment_2.py`](./run_experiment_2.py).

### Citing
Please cite:
~~~bibtex
@inproceedings{alaa2020generative,
  title={Generative Time-series Modeling with Fourier Flows},
  author={Alaa, Ahmed and Chan, Alex James and van der Schaar, Mihaela},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
~~~
