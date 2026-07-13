Work on:

https://github.com/evnekdev/matrixpacked

Create:

```text
agent/test-structured-factorizations
```

Commit and PR title:

```text
Test structured packed factorizations
```

# Goal

Verify packed Cholesky and packed Bunch-Kaufman workflows against nalgebra full matrices and residual-based oracle checks.

# Families

## Positive definite

```text
PackedSPD<f32>
PackedSPD<f64>
PackedSPD<Complex32>
PackedSPD<Complex64>
```

Test:

```text
PPTRF
PPTRS
PPTRI
PPCON
PPRFS
PPEQU
PPSV
PPSVX
```

only where currently implemented.

## Real symmetric indefinite

```text
PackedSymmetric<f32>
PackedSymmetric<f64>
```

Test:

```text
SPTRF
SPTRS
SPTRI
SPCON
SPRFS
SPSV
SPSVX
LANSP
```

where implemented.

## Complex symmetric

Audit separately.

Do not compare complex-symmetric operations using Hermitian nalgebra decompositions.

Use direct residual checks:

```text
A*x ≈ b
A*A_inv ≈ I
```

rather than a Hermitian oracle.

## Hermitian indefinite

```text
PackedHermitian<Complex32>
PackedHermitian<Complex64>
```

Test:

```text
HPTRF
HPTRS
HPTRI
HPCON
HPRFS
HPSV
HPSVX
LANHP
```

where implemented.

# Required test strategy

## 1. Factorization reconstruction

When feasible, reconstruct the logical factorization from returned packed factors and pivots.

For Cholesky:

```text
A ≈ L*L^H
A ≈ U^H*U
```

For Bunch-Kaufman, reconstruction is more complex due to pivots and 1×1/2×2 blocks.

Do not build an incorrect simplistic reconstruction.

Either:

* implement pivot-aware reconstruction carefully;
* or verify factors through solves and inverses.

A correct solve-based verification is better than a wrong factor reconstruction.

## 2. Solve

Generate known `X`.

Compute:

```text
B = A*X
```

using nalgebra.

Factorize and solve using `matrixpacked`.

Compare `X`.

Cover one and multiple RHS.

## 3. Inverse

Verify:

```text
A*A_inv ≈ I
A_inv*A ≈ I
```

For symmetric/Hermitian inverses, also verify returned structure.

## 4. Condition estimate

Compare qualitatively and within a reasonable factor to full reference condition numbers.

Use identity, diagonal, well-conditioned random, and intentionally ill-conditioned matrices.

## 5. Refinement

Perturb an initial solution and verify residual improvement.

## 6. Equilibration

Compare applied scaling against explicit full-matrix scaling:

```text
A_scaled(i,j) = s[i]*A(i,j)*s[j]
```

Verify positive scaling factors and diagonal behavior.

## 7. Simple and expert drivers

Compare results with the reusable factorization workflows.

For expert drivers, verify:

* solution residual;
* reciprocal condition number;
* forward/backward error vector lengths;
* equilibration outputs;
* ill-conditioned status behavior.

## 8. Failure cases

Test:

* non-positive-definite input to Cholesky;
* singular indefinite matrix;
* dimension mismatch;
* malformed RHS;
* mismatched original matrix and factorization in refinement;
* invalid supplied expert-driver options.

# Matrix generation

Use nalgebra to generate:

```text
SPD: A = M^T M + delta I
HPD: A = M^H M + delta I
symmetric indefinite: A = Q D Q^T
Hermitian indefinite: A = Q D Q^H
```

Ensure `D` contains controlled positive and negative eigenvalues.

For complex symmetric matrices, generate:

```text
A = M + M^T
```

without conjugation.

# Tests must not overfit storage bytes

The primary assertions must compare logical full matrices, solutions, inverses, and residuals.

Packed storage comparisons are supplemental.

# Validation

Run all relevant backends and inspect test runtime. Keep randomized cases bounded so normal `cargo test` remains practical.

Open a draft PR and finish only with:

**Safe to rebase and merge.**
