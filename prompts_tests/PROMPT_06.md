Work on:

https://github.com/evnekdev/matrixpacked

Create:

```text
agent/test-packed-eigensolvers
```

Commit and PR title:

```text
Test packed eigensolvers against nalgebra
```

# Goal

Verify all standard packed symmetric and Hermitian eigensolver APIs against nalgebra full-matrix decompositions and invariant-based checks.

# Operations

Test:

```text
SPEV
SPEVD
SPEVX
HPEV
HPEVD
HPEVX
```

and all corresponding public high-level methods.

# Critical comparison rules

Eigenvectors are not unique:

* real eigenvectors may differ by sign;
* complex eigenvectors may differ by unit complex phase;
* degenerate eigenspaces may have different orthonormal bases.

Therefore, never compare eigenvector entries directly except in carefully controlled simple-spectrum cases after phase alignment.

# Required assertions

## 1. Eigenvalues

For matrices with distinct eigenvalues:

* compare sorted packed results to nalgebra eigenvalues;
* use scale-aware tolerance;
* verify ascending order.

## 2. Residuals

For each returned eigenpair:

```text
||A*v - lambda*v||
```

must be small relative to:

```text
||A||*||v||
```

This is the primary correctness check.

## 3. Orthonormality/unitarity

For the returned eigenvector matrix `V`:

```text
V^T V ≈ I
V^H V ≈ I
```

as appropriate.

## 4. Reconstruction

Verify:

```text
A ≈ V*diag(lambda)*V^T
A ≈ V*diag(lambda)*V^H
```

for full eigendecompositions.

## 5. Comparison with nalgebra eigenspaces

For distinct eigenvalues, phase-align vectors before comparison:

```text
phase = dot(v_ref, v_actual) / |dot(...)|
```

For repeated or clustered eigenvalues, compare subspace projectors:

```text
P = V_cluster * V_cluster^H
```

rather than individual vectors.

## 6. Algorithm agreement

For the same matrix, compare outputs from:

```text
classical
divide-and-conquer
selected-all
```

Eigenvalues should agree within tolerance.

Eigenspaces should agree via projectors or residuals.

## 7. Selected ranges

Test:

* all;
* zero-based inclusive index ranges;
* value intervals;
* single selected eigenvalue;
* empty selection;
* invalid ranges;
* repeated eigenvalues near range boundaries.

Verify LAPACK’s value interval convention as implemented by the public API.

## 8. Ownership variants

Verify:

* borrowed method preserves original packed storage;
* consuming method may overwrite/reuse owned storage but returns correct result;
* mutable-view destructive method modifies backing storage according to documentation.

## 9. Edge cases

Test:

```text
n = 0
n = 1
n = 2
repeated eigenvalues
clustered eigenvalues
diagonal matrices
large dynamic range
```

# Nalgebra oracle

For real symmetric matrices, use nalgebra’s symmetric eigendecomposition.

For complex Hermitian matrices, confirm the selected nalgebra API supports Hermitian decomposition correctly. If nalgebra does not provide an appropriate direct complex-Hermitian decomposition for the version used, use invariant-based checks and compare eigenvalues through another independent full-storage approach already available in dev dependencies.

Do not introduce another native LAPACK wrapper as the “independent” oracle.

# Randomized testing

Use deterministic random orthogonal/unitary constructions with known spectra:

```text
A = Q*D*Q^T
A = Q*D*Q^H
```

This provides exact expected eigenvalues and controlled degeneracy.

# Validation

Keep matrices modest, such as up to `n = 16`, so test runtime stays reasonable.

Open a draft PR and finish only with:

**Safe to rebase and merge.**
