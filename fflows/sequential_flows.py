# Copyright (c) 2021, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""

This script contains the implementation for the spectral filter module of the Fourier flow

"""
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal

import fflows.logger as log
from fflows.filters.spectral import AttentionFilter, SpectralFilter
from fflows.fourier.transforms import DFT


class FourierFlow(nn.Module):
    def __init__(self, hidden, n_flows, FFT=True, flip=True, normalize=False):

        super().__init__()

        self.FFT = FFT
        self.normalize = normalize
        self.n_flows =n_flows
        self.hidden = hidden

        if flip:
            self.flips = [True if i % 2 else False for i in range(n_flows)]
        else:
            self.flips = [False for i in range(n_flows)]

    def forward(self, x):

        if self.FFT:
            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / (self.fft_std + 1e-8)

            x = x.view(-1, self.d + 1)

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):

            x, log_pz, lj = bijector(x, flip=f)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):

        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):

            z = bijector.inverse(z, flip=f)

        if self.FFT:

            if self.normalize:
                z = z * self.fft_std.view(-1, self.d + 1) + self.fft_mean.view(
                    -1, self.d + 1
                )

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):
        X_train = torch.from_numpy(np.array(X)).float()

        self.individual_shape = X_train.shape[1:]

        self.d = np.prod(self.individual_shape)
        self.k = int(np.floor(self.d / 2))

        # Prepare models
        self.bijectors = nn.ModuleList(
            [
                SpectralFilter(
                    self.d, self.k, self.FFT, hidden=self.hidden, flip=self.flips[_]
                )
                for _ in range(self.n_flows)
            ]
        )

        self.FourierTransform = DFT(N_fft=self.d)

        X_train = X_train.reshape(-1, self.d)

        # for normalizing the spectral transforms
        X_train_spectral = self.FourierTransform(X_train)[0]
        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        self.fft_std = torch.std(X_train_spectral, dim=0)
        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        all_epochs = int(np.floor(epochs / display_step))

        for step in range(epochs):

            optim.zero_grad()

            z, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()

            losses.append(loss.detach().numpy())

            loss.backward()
            optim.step()
            scheduler.step()

            if ((step % display_step) == 0) | (step == epochs - 1):

                current_epochs = int(np.floor((step + 1) / display_step))
                remaining_epochs = int(all_epochs - current_epochs)

                progress_signs = current_epochs * "|" + remaining_epochs * "-"
                display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"

                log.debug(display_string % (step, epochs, loss))

            if step == epochs - 1:
                log.debug("Finished training!")

        return losses

    def sample(self, n_samples):

        if self.FFT:

            mu, cov = torch.zeros(self.d + 1), torch.eye(self.d + 1)

        else:

            mu, cov = torch.zeros(self.d), torch.eye(self.d)

        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,))

        X_sample = self.inverse(z)

        return X_sample.reshape(-1, *self.individual_shape)


class RealNVP(nn.Module):
    def __init__(self, hidden, n_flows, flip=True, normalize=False):

        super().__init__()

        self.normalize = normalize
        self.hidden = hidden
        self.n_flows = n_flows
        self.FFT = False

        if flip:
            self.flips = [True if i % 2 else False for i in range(n_flows)]
        else:
            self.flips = [False for i in range(n_flows)]

    def forward(self, x):

        if self.FFT:

            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / (self.fft_std + 1e-8)

            x = x.view(-1, self.d + 1)

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):

            x, log_pz, lj = bijector(x, flip=f)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):

        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):

            z = bijector.inverse(z, flip=f)

        if self.FFT:

            if self.normalize:
                z = z * self.fft_std.view(-1, self.d + 1) + self.fft_mean.view(
                    -1, self.d + 1
                )

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):
        X_train = torch.from_numpy(np.array(X)).float()

        self.individual_shape = X_train.shape[1:]
        self.d = np.prod(self.individual_shape)
        self.k = int(np.floor(self.d / 2))

        self.bijectors = nn.ModuleList(
            [
                SpectralFilter(
                    self.d, self.k, self.FFT, hidden=self.hidden, flip=self.flips[_]
                )
                for _ in range(self.n_flows)
            ]
        )

        X_train = X_train.reshape(-1, self.d)

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        all_epochs = int(np.floor(epochs / display_step))

        for step in range(epochs):

            optim.zero_grad()

            z, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()

            losses.append(loss.detach().numpy())

            loss.backward()
            optim.step()
            scheduler.step()

            if ((step % display_step) == 0) | (step == epochs - 1):

                current_epochs = int(np.floor((step + 1) / display_step))
                remaining_epochs = int(all_epochs - current_epochs)

                progress_signs = current_epochs * "|" + remaining_epochs * "-"
                display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"

                log.debug(display_string % (step, epochs, loss))

            if step == epochs - 1:

                log.debug("Finished training!")

        return losses

    def sample(self, n_samples):

        if self.FFT:

            mu, cov = torch.zeros(self.d + 1), torch.eye(self.d + 1)

        else:

            mu, cov = torch.zeros(self.d), torch.eye(self.d)

        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,))

        X_sample = self.inverse(z)

        return X_sample.reshape(-1, *self.individual_shape)


class TimeFlow(nn.Module):
    def __init__(self, hidden, T, n_flows, flip=True, normalize=False):

        super().__init__()

        self.d = T
        self.k = int(T / 2) + 1
        self.normalize = normalize
        self.FFT = False

        if flip:

            self.flips = [True if i % 2 else False for i in range(n_flows)]

        else:

            self.flips = [False for i in range(n_flows)]

        self.bijectors = nn.ModuleList(
            [
                AttentionFilter(
                    self.d, self.k, self.FFT, hidden=hidden, flip=self.flips[_]
                )
                for _ in range(n_flows)
            ]
        )

    def forward(self, x):

        if self.FFT:

            x = self.FourierTransform(x)[0]

            if self.normalize:
                x = (x - self.fft_mean) / self.fft_std

            x = x.view(-1, self.d + 1)

        log_jacobs = []

        for bijector, f in zip(self.bijectors, self.flips):

            x, log_pz, lj = bijector(x, flip=f)

            log_jacobs.append(lj)

        return x, log_pz, sum(log_jacobs)

    def inverse(self, z):

        for bijector, f in zip(reversed(self.bijectors), reversed(self.flips)):

            z = bijector.inverse(z, flip=f)

        if self.FFT:

            if self.normalize:
                z = z * self.fft_std.view(-1, self.d + 1) + self.fft_mean.view(
                    -1, self.d + 1
                )

            z = self.FourierTransform.inverse(z)

        return z.detach().numpy()

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3, display_step=100):

        X_train = torch.from_numpy(np.array(X)).float()

        self.d = X_train.shape[1]
        self.k = int(np.floor(X_train.shape[1] / 2))

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        all_epochs = int(np.floor(epochs / display_step))

        for step in range(epochs):

            optim.zero_grad()

            z, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()

            losses.append(loss.detach().numpy())

            loss.backward()
            optim.step()
            scheduler.step()

            if ((step % display_step) == 0) | (step == epochs - 1):

                current_epochs = int(np.floor((step + 1) / display_step))
                remaining_epochs = int(all_epochs - current_epochs)

                progress_signs = current_epochs * "|" + remaining_epochs * "-"
                display_string = "step: %d \t/ %d \t" + progress_signs + "\tloss: %.3f"

                log.debug(display_string % (step, epochs, loss))

            if step == epochs - 1:

                log.debug("Finished training!")

        return losses

    def sample(self, n_samples):

        if self.FFT:

            mu, cov = torch.zeros(self.d + 1), torch.eye(self.d + 1)

        else:

            mu, cov = torch.zeros(self.d), torch.eye(self.d)

        p_Z = MultivariateNormal(mu, cov)
        z = p_Z.rsample(sample_shape=(n_samples,))

        X_sample = self.inverse(z)

        return X_sample
