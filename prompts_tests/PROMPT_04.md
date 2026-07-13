Work on:

https://github.com/evnekdev/matrixpacked

Create:

```text
agent/test-triangular-lapack-oracle
```

Commit and PR title:

```text
Test triangular LAPACK operations
```

# Goal

Verify every supported packed triangular operation against nalgebra full triangular matrices and residual-based checks.

# Operations

Audit and test all currently implemented triangular APIs, including:

```text
TPMV
TPSV
TPTRS
TPTRI
TPCON
TPRFS
LANTP
```

Do not add tests for `LATPS`, which is intentionally unsupported.

# Matrix coverage

Test:

```text
PackedLower
PackedUpper
```

For:

```text
f32
f64
Complex32
Complex64
```

For:

```text
Owned
View
ViewMut
```

where the operation permits it.

Use both:

```text
Diagonal::NonUnit
Diagonal::Unit
```

and:

```text
Transpose::None
Transpose::Transpose
Transpose::ConjugateTranspose
```

# Required tests

## 1. Single-vector solve

For each generated nonsingular triangular matrix:

1. generate known `x_expected`;
2. compute `b = op(A)*x_expected` in nalgebra;
3. call packed solve;
4. compare returned `x` to `x_expected`;
5. compute normalized residual.

## 2. Multi-RHS solve

Generate `X_expected` as a nalgebra matrix.

Compute:

```text
B = op(A)*X_expected
```

Store `B` in column-major layout.

Call the packed multi-RHS API.

Compare all columns.

Test at least:

```text
nrhs = 0, 1, 2, 4
```

according to API semantics.

## 3. Inverse

Compute packed inverse.

Expand to nalgebra.

Verify:

```text
A*A_inv ≈ I
A_inv*A ≈ I
```

Use both products because triangular orientation bugs may affect one path differently.

For unit-diagonal mode, construct the full logical matrix with unit diagonal before verification.

## 4. Norms

Compare:

```text
MaxAbs
One
Infinity
Frobenius
```

against independently calculated nalgebra/full-matrix norms.

Do not use nalgebra methods without checking their exact norm definitions.

Implement explicit oracle loops where necessary.

## 5. Reciprocal condition number

The LAPACK result is an estimate, so do not require equality with nalgebra.

Calculate a full reference condition number:

```text
cond_1(A) = ||A||_1 * ||A^-1||_1
cond_inf(A) = ||A||_inf * ||A^-1||_inf
```

Then test:

* returned value is finite and nonnegative;
* it is in a reasonable multiplicative range of `1/cond`;
* identity gives approximately `1`;
* strongly ill-conditioned matrices give much smaller values than well-conditioned matrices;
* singular or near-singular behavior follows documented semantics.

Avoid brittle exact tolerances for estimators.

## 6. Iterative refinement

Construct a deliberately perturbed solution.

Verify after refinement:

```text
residual_after <= residual_before
```

and usually:

```text
error_after <= error_before
```

Compare the refined solution with the known exact solution.

Check report lengths for all RHS columns.

Do not assume LAPACK’s `ferr` and `berr` exactly equal independently calculated errors.

## 7. Failure paths

Test:

* zero diagonal in non-unit mode;
* singular inverse;
* invalid RHS lengths;
* invalid increments;
* dimension overflow checks;
* malformed packed storage constructors;
* unit mode ignoring stored zero diagonal.

# Randomized cases

Use moderate well-conditioned random triangular matrices and targeted pathological matrices.

Keep random dimensions small enough for fast CI:

```text
1..12
```

# Validation

Run tests with each supported native backend feature used by CI.

If the repository supports both OpenBLAS and MKL CI, ensure test code is backend-independent.

Open a draft PR and finish only with:

**Safe to rebase and merge.**
