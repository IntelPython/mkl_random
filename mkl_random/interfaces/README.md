# Interfaces
The `mkl_random` package provides interfaces that serve as drop-in replacements for equivalent functions in NumPy.

---

## NumPy interface - `mkl_random.interfaces.numpy_random`

This interface is a drop-in replacement for the legacy portion of the [`numpy.random`](https://numpy.org/devdocs/reference/random/legacy.html) module and includes **all** classes and functions available there:

* random generator: `RandomState`.

* seeding and state functions: `get_state`, `set_state`, and `seed`.

* simple random data: `rand`, `randn`, `randint`, `random_integers`, `random_sample`, `choice` and `bytes`.

* permutations: `shuffle` and `permutation`

* distributions: `beta`, `binomial`, `chisquare`, `dirichlet`, `exponential`, `f`, `gamma`, `geometric`, `gumbel`, `hypergeometric`, `laplace`, `logistic`, `lognormal`, `multinomial`, `multivariate_normal`, `negative_binomial`, `noncentral_chisquare`, `noncentral_f`, `normal`, `pareto`, `poisson`, `power`, `rayleigh`, `standard_cauchy`, `standard_exponential`, `standard_gamma`, `standard_normal`, `standard_t`, `triangular`, `uniform`, `vonmises`, `wald`, `weibull`, and `zipf`.
