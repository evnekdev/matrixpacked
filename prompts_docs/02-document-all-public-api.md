# Prompt 02 — Document every public API item

Work on https://github.com/evnekdev/matrixpacked

Start only after Prompt 01 is merged.

## Workflow

Create `agent/document-public-api` from latest `master`. Open a draft PR titled **Document the public API**. Do not implement issue #41. Finish only with:

**Safe to rebase and merge.**

## Goal

Document every public module, type, trait, associated item, enum variant, public field, method, function, constant, alias, and error variant. Enforce completion with:

```rust
#![deny(missing_docs)]
```

## Inventory process

Temporarily use:

```rust
#![warn(missing_docs)]
```

Then run:

```bash
cargo check --lib --no-default-features
cargo check --lib --features nalgebra-interop
cargo doc --no-deps --features nalgebra-interop
```

Use compiler output as the primary inventory and manually inspect macro-generated public methods and re-exports.

The current API includes at least:

- `error`, `scalar`, `storage`, `triangular`, and `factorization`;
- lower, upper, symmetric, Hermitian, and SPD/HPD types and views;
- arithmetic, norms, rank updates, equilibration, refinement, and conditions;
- simple and expert solve APIs;
- inverse and diagnostic APIs (`Inertia`, `SignedLogDet`);
- standard, selected, divide-and-conquer, and generalized eigensolvers;
- generalized and tridiagonal reductions;
- LAPACK format conversions (`FullTriangular`, RFP types, `Triangle`, `RfpTranspose`);
- optional nalgebra conversions, validation tolerances, and conversion errors.

Do not rely solely on this list.

## Documentation standards

### Matrix families

Explain mathematical structure, supported scalars, packed ordering, stored triangle, mirroring/conjugation, structural zeros, packed length, and owned/view/view-mut forms.

Explicitly distinguish:

```text
PackedSymmetric<Complex<_>>: A^T = A
PackedHermitian<Complex<_>>: A^H = A
PackedSPD<Complex<_>>: HPD-intended structure
```

Do not claim positive definiteness is guaranteed unless the constructor validates it.

### Constructors and indexing

Document zero-based indexing, exact packed order, validation, copy/borrow behavior, mirrored access, mutation effects, errors, and panics.

### Numerical methods

State:

- mathematical equation;
- BLAS/LAPACK family where useful;
- vector/RHS memory layout;
- transpose/conjugate-transpose and unit-diagonal behavior;
- allocation and overwrite behavior;
- required factorization state;
- errors and numerical caveats.

### Factorizations

Explain represented factorization, pivots/block structure, ownership, destructive methods, post-`inverse_in_place` semantics, and determinant/inertia diagnostics.

### Condition and refinement

Explain reciprocal condition estimates, selected norm, `ferr`/`berr`, one entry per RHS, and requirements for original matrix plus factors.

### Solves

Explain reusable factors versus one-shot and expert drivers. Document column-major multi-RHS buffers and any unsupported supplied-factor modes.

### Eigen APIs

Document ascending eigenvalues, column-major eigenvectors, sign/phase ambiguity, repeated-eigenspace ambiguity, algorithm choices, selected ranges, generalized problem types, and normalization.

### Reductions

Document reflector storage, tridiagonal outputs, generated full Q, application side/operation, and generalized `ITYPE` transformations.

### Conversions

Separate:

1. LAPACK format conversion: TP ↔ full triangular and TP ↔ RFP.
2. Optional nalgebra conversion: allocation, extraction versus validation, tolerance rule, Cholesky-backed SPD/HPD validation, and lack of zero-copy views.

Describe the current provider requirement for triangular nalgebra conversion without making stable docs depend on GitHub issue numbers.

### Errors

Document every `PackedMatrixError` variant, its fields, cause, and likely corrective action.

## Required sections

Use where applicable:

```markdown
# Examples
# Errors
# Panics
# Safety
# Notes
```

Every public `Result` method should normally have `# Errors`; panic-capable methods need `# Panics`; unsafe APIs need `# Safety`.

Avoid boilerplate that merely repeats the method name.

## Links and source quality

Use intra-doc links. Do not copy Netlib prose verbatim. Use backticks for Rust generic notation. Ensure re-export links resolve.

## Doctests

Add representative doctests for:

- all matrix families;
- views and mutation;
- indexing/layout;
- arithmetic/multiplication/rank updates;
- factorization, solve, refinement, inverse, conditions;
- diagnostics;
- eigensolvers and generalized eigensolvers;
- format conversions;
- nalgebra extraction and strict validation.

Pure storage/nalgebra examples should run provider-free where possible. Use `no_run` for linked numerical examples only when appropriate; do not blanket-ignore doctests.

## Validation

```bash
cargo fmt --all
cargo check --all-targets --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo test --doc --features nalgebra-interop
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
git diff --check
```

Run linked forms where required.

Secondary audit:

```bash
rg '^\s*pub(\([^)]*\))?\s+(struct|enum|trait|fn|type|const|mod)\b' src
```

Review public macro expansions manually.

Do not change runtime behavior except for a tiny unavoidable documentation-discovered defect; otherwise open a follow-up issue.

The PR may use several logical commits, but no placeholders or undocumented public items may remain.
