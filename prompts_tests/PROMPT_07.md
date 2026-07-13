Work on:

https://github.com/evnekdev/matrixpacked

Create:

```text
agent/test-generalized-packed-eigensolvers
```

Commit and PR title:

```text
Test generalized packed eigensolvers
```

# Goal

Verify generalized symmetric/Hermitian-definite packed eigensolvers using nalgebra full matrices and invariant-based residual checks.

# Operations

Test:

```text
SPGV
SPGVD
SPGVX
HPGV
HPGVD
HPGVX
SPGST
HPGST
```

where implemented.

# Problem variants

Cover all LAPACK generalized problem types:

```text
A*x = lambda*B*x
A*B*x = lambda*x
B*A*x = lambda*x
```

Use the existing public enum; do not duplicate it.

# Matrix generation

Generate:

* symmetric/Hermitian `A`;
* SPD/HPD `B`.

For robust known-reference tests, construct controlled problems.

For type 1:

1. choose SPD/HPD `B`;
2. choose orthogonal/unitary `Q`;
3. choose real diagonal `D`;
4. construct an `A` whose transformed standard problem has spectrum `D`.

Alternatively, rely on residual and cross-algorithm agreement where exact construction is cumbersome.

# Required assertions

## 1. Generalized residual

For each eigenpair, verify the correct equation for the selected problem type.

For type 1:

```text
||A*v - lambda*B*v||
```

For types 2 and 3, verify their exact corresponding equations.

Normalize residuals appropriately.

## 2. Eigenvalue agreement

Compare:

* basic driver;
* divide-and-conquer;
* selected-all;
* nalgebra-transformed standard problem.

For type 1, use a full Cholesky transform with nalgebra:

```text
C = L^-1 A L^-H
```

Then compare eigenvalues of `C`.

Implement the transform carefully using solves, not explicit inverse when possible.

## 3. B-orthogonality

For type-1 eigenvectors, verify the appropriate generalized normalization, commonly:

```text
V^H B V ≈ I
```

Confirm LAPACK’s exact normalization convention for each problem type before asserting it.

## 4. Selected ranges

Test index and value selection with the same public semantics as standard eigensolvers.

## 5. Generalized reduction

For `SPGST`/`HPGST`, compare the reduced packed matrix against the independently calculated nalgebra Cholesky transformation.

## 6. Failure cases

Test:

* dimension mismatch;
* non-positive-definite `B`;
* invalid selection ranges;
* malformed packed storage;
* incompatible scalar or structure types where compile-time restrictions apply.

## 7. Mutation and allocation

Verify borrowed APIs preserve input matrices.

Verify consuming and mutable-view APIs follow documented overwrite behavior.

# Scalar coverage

Cover:

```text
f32
f64
Complex32
Complex64
```

for valid families.

# Degenerate spectra

Use projector/subspace comparisons rather than individual eigenvectors when eigenvalues repeat.

# Validation

Keep random dimensions moderate and deterministic.

Open a draft PR and finish only with:

**Safe to rebase and merge.**
