Work on:

https://github.com/evnekdev/matrixpacked

Create from the latest `master` after Prompt 2 is merged:

```text
agent/test-packed-arithmetic
```

Commit and PR title:

```text
Test packed arithmetic against nalgebra
```

# Goal

Verify all non-factorization arithmetic and BLAS-backed operations against nalgebra full-matrix calculations.

# Audit first

Search the public API for:

```text
add
sub
scale
mul
vector
rank1
rank2
assign
neg
axpy
```

Inspect all implemented arithmetic traits and methods.

Create an internal checklist before adding tests.

Do not test hypothetical APIs that do not exist.

# Test groups

## 1. Scalar arithmetic

For every supported matrix type and scalar:

* multiplying by a scalar;
* in-place scalar multiplication;
* division where supported;
* negation;
* zero and one coefficients;
* complex coefficients where mathematically valid.

Compare every logical entry against:

```rust
full_matrix * scalar
```

in nalgebra.

Verify structure remains valid:

* triangular remains triangular;
* symmetric remains symmetric;
* Hermitian remains Hermitian only for valid scalar operations;
* SPD invariants are not falsely preserved after arbitrary scaling.

If the crate restricts scalar multiplication on Hermitian/SPD types, test those restrictions.

## 2. Addition and subtraction

Where supported, verify:

```text
A + B
A - B
A += B
A -= B
```

against nalgebra.

Check dimension mismatch errors.

Check structural compatibility.

Do not assume adding two SPD matrices through a generic API exists; audit actual methods.

## 3. Packed matrix-vector multiplication

Test:

```text
TPMV
SPMV
HPMV
```

For:

* lower and upper triangular;
* symmetric real;
* complex symmetric if supported by an operation;
* Hermitian complex;
* SPD/HPD;
* `f32`, `f64`, `Complex32`, `Complex64`;
* owned, view, and mutable view where relevant.

Compare with nalgebra:

```rust
expected = full * vector
```

## 4. Transpose modes

For triangular operations, compare:

```text
A*x
A^T*x
A^H*x
```

against nalgebra.

For real types, verify transpose and conjugate transpose agree.

For complex types, construct matrices where transpose and conjugate transpose differ significantly.

## 5. Unit diagonal

For triangular methods accepting `Diagonal::Unit`, ensure nalgebra oracle replaces the stored diagonal logically with ones before multiplication or solve.

Include deliberately non-one stored diagonal entries to prove the option is honored.

## 6. Strided vectors

Test positive strides such as:

```text
1, 2, 3
```

and negative strides if supported by the public contract.

Construct the logical vector independently from the physical buffer.

Verify:

* untouched padding values remain unchanged;
* only logical vector elements are modified;
* insufficient backing lengths are rejected;
* zero stride is rejected.

## 7. Rank updates

After rank-update APIs exist, verify:

```text
SPR
SPR2
HPR
HPR2
```

against nalgebra full updates.

For Hermitian updates:

```text
A + alpha*x*x^H
A + alpha*x*y^H + conj(alpha)*y*x^H
```

Verify:

* logical result;
* conjugate symmetry;
* real diagonal;
* strided vector handling;
* mutable view behavior.

## 8. Allocating versus in-place API equivalence

Where both exist:

```text
operation(...)
operation_in_place(...)
```

verify they produce identical logical matrices or vectors.

# Matrix sizes

Use deterministic sizes:

```text
0, 1, 2, 3, 5, 8
```

plus property-based dimensions up to a modest size.

# Numerical comparison

For direct arithmetic and BLAS Level-2 operations, use tighter tolerances than for factorization.

Suggested starting defaults:

```text
f32: abs 1e-5, rel 1e-4
f64: abs 1e-12, rel 1e-11
```

Adjust based on dimension and operation count, not ad hoc per failing test.

# Validation

Run all test suites and ensure test names are granular enough to identify the failing matrix family and scalar.

Open a draft PR and end only with:

**Safe to rebase and merge.**
