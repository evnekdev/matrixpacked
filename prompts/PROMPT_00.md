Work on the public GitHub repository:

https://github.com/evnekdev/matrixpacked

Create a new branch from the latest `master`.

This task must be performed **after** the custom `LATPS` removal PR has been merged.

---

# Goal

Implement the complete packed-storage **condition number estimation** functionality for every matrix family supported by LAPACK.

This is Priority 1 because:

* it naturally complements factorization;
* requires no new storage structures;
* is available directly in LAPACK;
* has a clean API;
* is useful in virtually every numerical workflow.

No handwritten FFI.

Use only:

* Rust `lapack`
* Rust `blas`

---

# Scope

Implement only:

## Triangular

```text
xTPCON
```

already present

Review it carefully.

Verify correctness.

Improve API if necessary.

---

## SPD

```text
xPPCON
```

Implement for

```
f32
f64
Complex32
Complex64
```

---

## Symmetric indefinite

```text
xSPCON
```

Implement for

```
f32
f64
Complex32
Complex64
```

---

## Hermitian indefinite

```text
xHPCON
```

Implement for

```
Complex32
Complex64
```

---

# API philosophy

Mirror the existing style.

For every factorization object expose

```rust
condition_number(...)
```

or

```rust
rcond(...)
```

Choose whichever is already more consistent inside the crate.

I personally slightly prefer

```rust
rcond()
```

because LAPACK computes reciprocal condition numbers.

---

# Avoid allocations

The implementation should avoid unnecessary copying.

Workspace vectors should only be allocated if LAPACK requires them.

Prefer

```
Vec<Scalar::Real>
```

or

```
Vec<i32>
```

exactly matching LAPACK requirements.

Do not allocate matrices.

---

# Factorization types

Condition estimation should only exist for already-factorized matrices.

Examples:

```rust
PackedSPDFactor
PackedSymmetricFactor
PackedHermitianFactor
PackedTriangular
```

(or whatever the current naming convention is)

Do NOT recompute the factorization internally.

The caller should already have

```
factor = matrix.factorize()?
```

then

```
let rcond = factor.rcond()?;
```

---

# Matrix norm

Most LAPACK condition routines require

```
||A||
```

Do NOT recompute it internally every time if the API already has a norm routine.

Investigate whether

```
matrix_norm(...)
```

already exists.

Reuse it.

If useful, expose

```rust
factor.rcond_with_norm(norm)
```

to avoid repeated norm calculations.

---

# Error handling

Reuse existing error types.

No panics.

---

# Examples

Generate examples for every scalar type.

Examples should follow existing project style.

---

## Triangular

already present

Review only.

---

## SPD

Generate

```
lapack_spd_f32_ppcon.rs
lapack_spd_f64_ppcon.rs
lapack_spd_c32_ppcon.rs
lapack_spd_c64_ppcon.rs
```

---

## Symmetric

Generate

```
lapack_symmetric_f32_spcon.rs
lapack_symmetric_f64_spcon.rs
lapack_symmetric_c32_spcon.rs
lapack_symmetric_c64_spcon.rs
```

---

## Hermitian

Generate

```
lapack_hermitian_c32_hpcon.rs
lapack_hermitian_c64_hpcon.rs
```

---

Each example should

1.

construct a tiny matrix

2.

factorize

3.

compute

```
rcond
```

4.

compare against expected value

within tolerance

---

# Documentation

Update

```
PACKED_LAPACK_FUNCTIONS.md
```

mark

```
TPCON
PPCON
SPCON
HPCON
```

implemented

Update

```
EXAMPLE_COVERAGE.md
```

with new example count.

---

# Validation

Run

```
cargo fmt
cargo check
cargo check --examples
```

Inspect

```
git diff --stat
git diff
```

No unrelated changes.

---

# Commit

Single commit

```
Implement packed condition estimation
```

---

# Pull request

Title

```
Implement packed condition estimation
```

Description

Summarize

* PPCON
* SPCON
* HPCON
* reviewed TPCON
* examples
* documentation

Do not stop after partial implementation.

Only finish after:

* every scalar type implemented
* every example added
* documentation updated
* examples compile
* no TODOs remain

Then finish your response with exactly

**Safe to rebase and merge.**
