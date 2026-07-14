# Prompt 03 — Build comprehensive crate-level documentation

Work on https://github.com/evnekdev/matrixpacked

Start only after Prompt 02 is merged.

Create `agent/comprehensive-crate-docs`. Open a draft PR titled **Add comprehensive crate documentation**. Finish only with:

**Safe to rebase and merge.**

## Context

The nalgebra interoperability series already added a nalgebra guide, README material, examples, and docs.rs feature configuration. The oracle series added testing documentation and CI. Preserve and integrate those materials; do not duplicate or overwrite them.

## Goal

Make the Rustdoc landing page and repository documentation a coherent learning guide. Item-level missing docs were handled in Prompt 02; this PR focuses on navigation, concepts, and end-to-end workflows.

## Documentation architecture

Audit:

- crate-level docs in `src/lib.rs`;
- `README.md`;
- `docs/`;
- `NALGEBRA_INTEROP.md`;
- `TESTING.md`;
- `PACKED_LAPACK_FUNCTIONS.md`;
- `EXAMPLE_COVERAGE.md`;
- examples and scripts.

Choose one source of truth for long-form crate docs, for example:

```rust
#![doc = include_str!("../docs/crate.md")]
```

Do not blindly include the README if badges or repository-relative links render poorly in Rustdoc. Avoid maintaining duplicate long guides.

## Required guide sections

1. **Purpose and design**
   - traditional packed storage;
   - `n(n+1)/2` memory;
   - direct packed BLAS/LAPACK;
   - limitations versus dense Level-3 operations.

2. **Matrix family map**
   - lower, upper, symmetric, Hermitian, SPD/HPD;
   - structure, scalars, stored data, logical behavior.

3. **Exact packed layout**
   - lower and upper packed-column formulas;
   - 3×3 coordinate examples;
   - logical versus physical entries.

4. **Ownership model**
   - owned `Vec<T>`;
   - immutable and mutable views;
   - zero-copy operations;
   - destructive LAPACK calls;
   - borrowed-copy policy;
   - dense/nalgebra allocation cost.

5. **Basic operations**
   - construction, access, mutation, arithmetic, matrix-vector multiplication, rank updates.

6. **Linear systems**
   - reusable factorization;
   - simple one-shot solve;
   - expert solve;
   - iterative refinement;
   - when to choose each.

7. **Inverse, norms, conditions, diagnostics**
   - prefer solves over explicit inverse;
   - reciprocal condition estimates;
   - determinant/log determinant;
   - inertia and singularity.

8. **Eigenproblems**
   - standard, divide-and-conquer, selected, generalized;
   - low-level reductions;
   - vector sign/phase and repeated-subspace ambiguity.

9. **Conversions and interoperability**
   - TP/full triangular/RFP;
   - optional nalgebra feature;
   - conversion rather than views;
   - extraction versus strict validation;
   - SPD/HPD validation;
   - current triangular nalgebra provider requirement (issue #41 remains deferred).

10. **Native providers**
    - Rust crates are bindings;
    - OpenBLAS/Linux and MKL/Windows strategy;
    - provider feature selection;
    - no inaccurate Windows OpenBLAS claims;
    - docs.rs behavior.

11. **Testing**
    - unit/integration;
    - nalgebra oracle;
    - deterministic property tests;
    - extended stress tier;
    - reproducible seeds and backend-qualified commands.

12. **Errors and numerical behavior**
    - approximate arithmetic;
    - estimator behavior;
    - ill-conditioning;
    - singular/positive-definite failures.

13. **Feature and support tables**
    - scalar and matrix family support;
    - optional features;
    - links to detailed routine status instead of duplicating it.

## Module pages

Review every public module’s `//!` landing text. Each should explain purpose, principal types, a common workflow, and links to related modules/crate guide.

## README

Keep it concise:

- purpose;
- supported structures;
- short example;
- backend requirement;
- optional nalgebra feature;
- testing/docs commands;
- links to detailed guides/status.

## Examples

Verify examples use current APIs and feature gates. Document runner scripts accurately. Do not commit generated `target/doc` or binary screenshots.

## Validation

```bash
cargo fmt --all -- --check
cargo check --all-targets --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo test --doc --features nalgebra-interop
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
git diff --check
```

Run provider-linked examples/tests where needed.

Manually inspect generated Rustdoc for landing page, navigation, module pages, links, tables, code rendering, feature-gated items, and narrow-window readability.

PR description: list the guide structure, module/README changes, integrated existing nalgebra/testing material, doctest status, and manual inspection.
