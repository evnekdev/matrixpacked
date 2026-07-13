Work on:

https://github.com/evnekdev/matrixpacked

Start after the public-API documentation PR has been merged.

Create:

```text
agent/comprehensive-crate-docs
```

Commit and PR title:

```text
Add comprehensive crate documentation
```

# Goal

Turn the generated Rustdoc landing page into a coherent user guide for `matrixpacked`.

The current crate-level documentation is only a short description and backend-linking note. Expand it into a practical guide without duplicating every method’s API documentation.

# Crate-level structure

Expand `src/lib.rs` crate documentation using included Markdown where appropriate.

Preferred structure:

```rust
#![doc = include_str!("../README.md")]
```

only if the README is designed to compile cleanly as Rustdoc and does not contain unsuitable repository-only content.

A cleaner option may be:

```rust
#![doc = include_str!("../docs/crate.md")]
```

with README remaining concise.

Choose the structure that avoids duplicated content and broken links.

# Required guide sections

## 1. What the crate provides

Explain traditional packed storage and why it exists:

* square structured matrices;
* `n(n+1)/2` stored elements;
* reduced memory use;
* direct packed BLAS/LAPACK operations;
* limitations compared with dense Level-3 BLAS.

## 2. Matrix families

Provide a comparison table:

| Type              | Mathematical structure | Scalars      | Stored data    |
| ----------------- | ---------------------- | ------------ | -------------- |
| `PackedLower`     | lower triangular       | real/complex | lower triangle |
| `PackedUpper`     | upper triangular       | real/complex | upper triangle |
| `PackedSymmetric` | `Aᵀ = A`               | real/complex | one triangle   |
| `PackedHermitian` | `Aᴴ = A`               | complex      | one triangle   |
| `PackedSPD`       | SPD/HPD                | real/complex | one triangle   |

Clearly explain that complex symmetric and Hermitian matrices are different.

## 3. Traditional packed layout

Document exact BLAS packed-column ordering.

For lower storage:

```text
(a00, a10, a20, ..., a(n-1)0, a11, a21, ...)
```

For upper storage:

```text
(a00, a01, a11, a02, a12, a22, ...)
```

Verify these formulas against the actual crate before publishing.

Provide diagrams or coordinate tables for `3 × 3` examples.

## 4. Owned, view, and mutable view types

Explain:

* owned `Vec<T>` storage;
* immutable borrowed slice;
* mutable borrowed slice;
* zero-copy mutation;
* lifetime behavior;
* which operations require writable storage;
* which operations allocate packed copies.

Include examples.

## 5. Basic operations

Show:

* construction;
* logical indexing;
* raw packed storage access;
* scalar arithmetic;
* matrix-vector multiplication;
* rank updates.

## 6. Factorization and solves

Show preferred workflows:

```rust
let factor = matrix.factorize()?;
let x = factor.solve_vector(&b)?;
```

Explain when to use:

* reusable factorization;
* one-shot solve;
* expert solve;
* iterative refinement.

## 7. Inverses and condition estimates

Explain:

* prefer solving to explicitly computing an inverse;
* reciprocal condition number;
* norm selection;
* ownership behavior of inverse methods.

## 8. Eigenvalue APIs

Explain:

* standard symmetric/Hermitian problems;
* basic versus divide-and-conquer;
* selected eigenvalues;
* generalized eigenproblems;
* eigenvector layout;
* sign/phase ambiguity.

## 9. Backend linking

Document supported features accurately:

```text
openblas-static
intel-mkl-static
```

Explain:

* Rust `blas`/`lapack` crates provide bindings;
* a native provider is still required;
* platform considerations;
* feature selection;
* why provider features should generally be enabled by the final binary.

Do not claim `openblas-static` works on all Windows configurations if it does not.

## 10. Error handling

Show idiomatic `Result` handling and describe common errors:

* invalid packed length;
* dimension mismatch;
* singular matrix;
* non-positive-definite factorization;
* LAPACK illegal argument;
* invalid selection range.

## 11. Feature and scalar support table

Create a maintained table mapping major operations to:

```text
f32
f64
Complex32
Complex64
```

Do not duplicate the entire internal roadmap.

Link to `PACKED_LAPACK_FUNCTIONS.md` for comprehensive routine status if appropriate.

## 12. Numerical behavior

Document:

* floating-point tolerances;
* no exact equality expectation;
* condition estimation is approximate;
* eigenvector non-uniqueness;
* packed storage performance trade-offs;
* destructive LAPACK calls and crate copying policy.

# Module-level documentation

Add substantial `//!` documentation to public modules where crate-level material would be too broad.

Each public module should explain its purpose and link to primary exported types.

# Examples

All guide code examples must compile as doctests where reasonable.

Separate examples that require native linking from pure storage examples.

Do not include stale APIs copied from old examples.

# README alignment

Update the README so a GitHub visitor sees:

* project purpose;
* supported matrix structures;
* short example;
* backend requirements;
* testing command;
* documentation command;
* links to API status and example coverage.

Avoid maintaining two independently divergent long guides.

# Documentation assets

Do not add generated `target/doc` HTML to Git.

Do not commit large binary screenshots when text/Markdown diagrams suffice.

# Validation

Run:

```bash
cargo fmt --all
cargo test
cargo test --doc
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo clippy --all-targets -- -D warnings
git diff --check
```

Open the generated documentation locally and manually inspect:

* landing page;
* module navigation;
* intra-doc links;
* code formatting;
* tables;
* mobile-width readability if practical.

The PR description must list the guide sections added and doctest status.

Open a draft PR and finish only with:

**Safe to rebase and merge.**
