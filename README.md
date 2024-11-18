# Tanimoto GP

Work in progress- a small library for Tanimoto kernel GPs.

For a minimal running example, see `examples/logp_regression.py`.

## Development

### Installation

Install `kern_gp` using:

```bash
https://github.com/AustinT/kernel-only-GP.git
```

(currently works with v0.1.0, future compatibility unclear)

### Formatting

Use pre-commit to enforce formatting, large file checks, etc.

If not already installed in your environment, run:

```bash
conda install pre-commit
```

To install the precommit hooks:

```bash
pre-commit install
```

Now a series of useful checks will be run before any commit.

## List of possible improvements

- Cache the cholesky factorization of kernel matrix to avoid repeatedly computing this in BO
