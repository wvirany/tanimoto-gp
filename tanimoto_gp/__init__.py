from typing import Any, Callable, NamedTuple

import kern_gp as kgp
from jax import numpy as jnp
from jax.nn import softplus
from rdkit import DataStructs

TRANSFORM = softplus  # fixed transform function


class TanimotoGP_Params(NamedTuple):
    # Inverse softplus of GP parameters
    raw_amplitude: jnp.ndarray
    raw_noise: jnp.ndarray

    # Consant mean function
    mean: jnp.ndarray


class ConstantMeanTanimotoGP:
    def __init__(self, fp_func: Callable[[str], Any], smiles_train: list[str], y_train):
        super().__init__()
        self._fp_func = fp_func
        self.set_training_data(smiles_train, y_train)

    def set_training_data(self, smiles_train: list[str], y_train: jnp.ndarray):
        self._smiles_train = smiles_train
        self._y_train = jnp.asarray(y_train)
        self._fp_train = [self._fp_func(smiles) for smiles in smiles_train]
        self._K_train_train = jnp.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in self._fp_train]
        )

    def marginal_log_likelihood(self, params: TanimotoGP_Params) -> jnp.ndarray:
        y_centered = self._y_train - params.mean
        return kgp.mll_train(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._K_train_train,
            y_train=y_centered,
        )

    def predict_f(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True) -> jnp.ndarray:

        # Construct kernel matrices
        fp_test = [self._fp_func(smiles) for smiles in smiles_test]
        K_test_train = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in fp_test])
        if full_covar:
            K_test_test = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, fp_test) for fp in fp_test])
        else:
            K_test_test = jnp.ones((len(smiles_test)), dtype=float)

        # Get centered predictions
        mean = params.mean
        y_centered = self._y_train - mean

        # Get predictions from centered GP
        mean_pred, covar = kgp.noiseless_predict(
            a=TRANSFORM(params.raw_amplitude),
            s=TRANSFORM(params.raw_noise),
            k_train_train=self._K_train_train,
            k_test_train=K_test_train,
            k_test_test=K_test_test,
            y_train=y_centered,
            full_covar=full_covar,
        )

        # Add mean back to predictions
        return mean_pred + mean, covar

    def predict_y(self, params: TanimotoGP_Params, smiles_test: list[str], full_covar: bool = True) -> jnp.ndarray:
        mean, covar = self.predict_f(params, smiles_test, full_covar)
        if full_covar:
            covar = covar + jnp.eye(len(smiles_test)) * TRANSFORM(params.raw_noise)
        else:
            covar += TRANSFORM(params.raw_noise)
        return mean, covar


class FixedTanimotoGP:
    """ConstantMeanTanimotoGP with fixed parameters and test set for caching data"""

    def __init__(
        self,
        fp_func: Callable[[str], Any],
        smiles_train: list[str],
        y_train: jnp.ndarray,
        amplitude: float,
        noise: float,
        mean: float = 0.0,
    ):
        if amplitude <= 0:
            raise ValueError("Amplitude must be positive")
        if noise <= 0:
            raise ValueError("Noise must be positive")

        self._fp_func = fp_func
        self._amplitude = amplitude
        self._noise = noise
        self._mean = mean
        self.set_training_data(smiles_train, y_train, mean)

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @property
    def noise(self) -> float:
        return self._noise

    @property
    def mean(self) -> float:
        return self._mean

    def set_training_data(self, smiles_train: list[str], y_train: jnp.ndarray, mean: float = 0.0):
        self._smiles_train = smiles_train
        self._y_train = jnp.asarray(y_train)
        self._y_centered = self._y_train - mean
        self._fp_train = [self._fp_func(smiles) for smiles in smiles_train]
        self._K_train_train = jnp.asarray(
            [DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in self._fp_train]
        )
        self._cached_L = kgp._k_cholesky(self._K_train_train, self._noise / self._amplitude)
        self._K_test_train = None
        self._smiles_test = None

    def marginal_log_likelihood(self) -> jnp.ndarray:
        return kgp.mll_train(
            a=self._amplitude,
            s=self._noise,
            k_train_train=self._K_train_train,
            y_train=self._y_centered,
        )

    def _get_predictions(self, k_test_train, k_test_test, full_covar: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get centered predictions (before adding back mean)"""
        if self._cached_L is not None:
            return kgp._L_noiseless_predict(
                a=self._amplitude,
                L=self._cached_L,
                k_test_train=k_test_train,
                k_test_test=k_test_test,
                y_train=self._y_centered,
                full_covar=full_covar,
            )
        else:
            return kgp.noiseless_predict(
                a=self._amplitude,
                s=self._noise,
                k_train_train=self._K_train_train,
                k_test_train=k_test_train,
                k_test_test=k_test_test,
                y_train=self._y_centered,
                full_covar=full_covar,
            )

    def predict_f(
        self, smiles_test: list[str], full_covar: bool = True, from_train: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        # Handle predictions for training points
        if from_train:
            K_test_test = self._K_train_train if full_covar else jnp.ones(len(smiles_test), dtype=float)
            mean_centered, covar = self._get_predictions(self._K_train_train, K_test_test, full_covar)
            return mean_centered + self._mean, covar

        # Initialize K_test_train and smiles_test if needed
        if self._K_test_train is None:
            fp_test = [self._fp_func(smiles) for smiles in smiles_test]
            self._K_test_train = jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, self._fp_train) for fp in fp_test])
            self._smiles_test = smiles_test

        fp_test = [self._fp_func(smiles) for smiles in self._smiles_test]

        # Compute K_test_test
        K_test_test = (
            jnp.asarray([DataStructs.BulkTanimotoSimilarity(fp, fp_test) for fp in fp_test])
            if full_covar
            else jnp.ones(len(smiles_test), dtype=float)
        )

        mean_centered, covar = self._get_predictions(self._K_test_train, K_test_test, full_covar)
        return mean_centered + self._mean, covar

    def predict_y(
        self, smiles_test: list[str], full_covar: bool = True, from_train: bool = False
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        mean, covar = self.predict_f(smiles_test, full_covar, from_train)
        if full_covar:
            covar = covar + jnp.eye(len(smiles_test)) * self._noise
        else:
            covar = covar + self._noise
        return mean, covar

    def add_observations(self, idx: int, new_y: float):
        """
        Adds a single observation and efficiently updates cached matrices

        Args:
            idx: Index in test set of new observation
            new_y: Value of new observation
        """

        # Get SMILES and kernel row for new point
        new_smiles = self._smiles_test[idx]
        k_new = self._K_test_train[idx]
        new_fp = self._fp_func(new_smiles)

        # Update training data
        self._smiles_train.append(new_smiles)
        self._y_train = jnp.append(self._y_train, new_y)
        self._y_centered = self._y_train - self._mean
        self._fp_train.append(new_fp)

        # Update K_train_train w/ row from K_test_train
        n = len(k_new)
        top_block = jnp.concatenate([self._K_train_train, k_new.reshape(n, 1)], axis=1)
        bottom_row = jnp.concatenate([k_new, jnp.array([1.0])])
        self._K_train_train = jnp.concatenate([top_block, bottom_row.reshape(1, n + 1)], axis=0)

        # Remove selected point from test set matrices
        self._K_test_train = jnp.delete(self._K_test_train, idx, axis=0)
        self._smiles_test.pop(idx)

        # Compute similarities between new point and remaining test points
        remaining_fps = [self._fp_func(s) for s in self._smiles_test]
        k_new_test = jnp.asarray(DataStructs.BulkTanimotoSimilarity(new_fp, remaining_fps))

        # Add new column to K_test_train
        self._K_test_train = jnp.column_stack([self._K_test_train, k_new_test])

        # If we have cached Cholesky factor, update it efficiently
        if self._cached_L is not None:
            self._cached_L = kgp.update_cholesky(L=self._cached_L, k_new=k_new, k_new_new=1.0)
