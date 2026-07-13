Work on:

https://github.com/evnekdev/matrixpacked

Start only after the warning-cleanup PR has been merged.

Create a fresh branch from the latest `master`:

```text
agent/document-public-api
```

Commit and PR title:

```text
Document the public API
```

# Goal

Add complete, accurate Rustdoc documentation for every public API item in the crate.

All public code items must be documented, including:

* public modules;
* structs;
* enums;
* enum variants;
* public fields;
* traits;
* associated types;
* trait methods;
* functions;
* methods;
* type aliases;
* constants;
* re-exported public types where documentation is otherwise difficult to find;
* public error variants.

The documentation must explain behavior, not merely restate the identifier.

# Enforcement strategy

At the beginning of the task, temporarily add:

```rust
#![warn(missing_docs)]
```

to `src/lib.rs`.

Run:

```bash
cargo check --lib
cargo doc --no-deps
```

Use the warnings as the complete documentation inventory.

Do not immediately use:

```rust
#![deny(missing_docs)]
```

while editing.

At the end, once every warning is resolved, promote it to:

```rust
#![deny(missing_docs)]
```

unless generated code or an unavoidable external limitation makes that unreasonable.

Do not use blanket `#[allow(missing_docs)]`.

# Documentation quality requirements

Every public item should answer the applicable questions below.

## Matrix types

Document:

* mathematical structure;
* storage layout;
* packed length;
* whether the stored triangle is lower or upper;
* symmetry or conjugation behavior;
* ownership/storage parameter;
* whether the type enforces structural or numerical invariants;
* relationship among owned, view, and mutable-view aliases.

For example, clearly distinguish:

```text
PackedSymmetric<Complex<_>>: Aᵀ = A
PackedHermitian<Complex<_>>: Aᴴ = A
PackedSPD<Complex<_>>: Hermitian positive definite
```

## Constructors

Document:

* expected packed ordering;
* required storage length;
* whether input is copied or borrowed;
* validation performed;
* error conditions;
* examples for non-obvious layouts.

Include packed-column formulas or links to crate-level layout documentation rather than leaving users to infer them.

## Access methods

Document:

* zero-based indexing;
* logical versus physically stored entries;
* mirrored access;
* conjugation behavior;
* behavior outside the stored triangle;
* panic versus `Result` behavior;
* mutation effects on mirrored entries.

## Numerical operations

Document:

* mathematical equation;
* BLAS/LAPACK family used;
* whether input storage is overwritten;
* allocation behavior;
* vector and matrix layout;
* transpose/conjugate-transpose semantics;
* unit-diagonal semantics;
* required factorization state;
* error conditions;
* numerical caveats.

For example, a solve method should state:

```text
Solves op(A)X = B.
```

and identify:

* column-major RHS layout;
* `nrhs`;
* leading dimension assumptions;
* whether the factorization is reused;
* whether `A`, factors, or RHS are modified.

## Factorization objects

Document:

* exact factorization represented;
* packed storage contents after factorization;
* pivot representation where applicable;
* whether the object owns or borrows storage;
* methods that consume or invalidate factorization semantics;
* behavior after `inverse_in_place()`.

## Eigenvalue APIs

Document:

* ascending eigenvalue order;
* eigenvector column-major layout;
* sign/phase non-uniqueness;
* destructive versus borrowing behavior;
* range conventions;
* zero-based Rust indices versus one-based LAPACK indices;
* value interval inclusivity;
* generalized problem type.

## Result and options structures

Document every public field.

Explain units, lengths, and interpretation.

Examples:

* reciprocal condition number;
* forward and backward error vectors;
* scaling factors;
* selected eigenvalue count;
* eigenvector layout;
* equilibration state.

## Errors

Document every `PackedMatrixError` variant:

* triggering condition;
* meaning of stored fields;
* likely corrective action.

## Unsafe public APIs

If any exist, include a proper:

```markdown
# Safety
```

section describing all caller obligations.

# Required Rustdoc sections

Use these sections where applicable:

```markdown
# Examples
# Errors
# Panics
# Safety
# Notes
```

Do not add empty or boilerplate sections.

Every public method returning `Result` should normally have an `# Errors` section unless the errors are truly self-evident and already documented through a directly linked common contract.

Every public method that may panic should have `# Panics`.

# Intra-doc links

Use Rustdoc intra-doc links:

```rust
[`PackedLower`]
[`PackedStorage`]
[`factorize`](Self::factorize)
```

Do not use fragile plain-text references when a valid link is possible.

Run rustdoc with broken-link warnings denied.

# Examples and doctests

Add small compiling examples to the most important APIs:

* create each packed matrix family;
* create views;
* logical indexing;
* matrix-vector multiplication;
* factorize and solve;
* inverse;
* condition estimation;
* refinement;
* rank update;
* eigenvalues/eigenvectors;
* generalized eigenproblem.

Do not duplicate huge examples on every method.

Use crate-level and module-level examples for workflows, and focused method-level examples for subtle behavior.

Doctests must not depend on an unavailable native backend unless marked appropriately.

For operations requiring BLAS/LAPACK linking, decide on a sustainable approach:

* use `no_run` only when compilation is still meaningful;
* avoid `ignore` unless there is a documented platform reason;
* keep pure storage examples executable.

Do not make all doctests `ignore`.

# Modules to inspect

At minimum:

```text
src/lib.rs
src/error.rs
src/scalar.rs
src/storage.rs
src/triangular.rs
src/factorization.rs
src/lower.rs
src/upper.rs
src/symmetric.rs
src/hermitian.rs
src/spd.rs
src/eigen.rs
src/equilibration.rs
src/expert_solve.rs
src/tridiagonal.rs
```

Also inspect public items generated by macros in:

```text
src/arithmetic.rs
src/norms.rs
src/rank_updates.rs
src/simple_solve.rs
```

Macro-generated public methods still require useful documentation.

# Avoid low-quality documentation

Do not write documentation such as:

```rust
/// Returns the dimension.
pub fn dimension(...)
```

Prefer:

```rust
/// Returns the number of logical rows and columns in the square matrix.
```

Do not paste raw Netlib documentation verbatim.

Summarize semantics in original wording and link concepts through Rustdoc.

Do not claim stronger invariants than the type actually enforces.

In particular, confirm whether `PackedSPD` validates positive definiteness at construction or merely represents a matrix intended to be SPD/HPD.

# Validation

Run:

```bash
cargo fmt --all
cargo check --all-targets
cargo test
cargo test --doc
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --document-private-items
cargo clippy --all-targets -- -D warnings
git diff --check
```

Also verify:

```bash
cargo rustdoc --lib -- -D rustdoc::broken_intra_doc_links
```

Use equivalent stable commands if direct rustdoc flags differ on the installed toolchain.

Search for undocumented public declarations manually as a second check:

```bash
rg '^\s*pub(\([^)]*\))?\s+(struct|enum|trait|fn|type|const|mod)\b' src
```

Do not rely solely on `missing_docs`; public fields and macro expansions need careful review.

# Review size

This will be a large documentation PR.

Organize commits logically if necessary:

1. storage and matrix types;
2. operations and factorizations;
3. eigen/advanced APIs;
4. errors and supporting types;
5. enforcement.

The final PR may contain multiple well-scoped commits, but no unfinished sections or placeholder text.

Do not change API behavior unless fixing a clear documentation-discovered bug. Put behavioral changes into separate follow-up PRs.

The PR description must summarize:

* public modules documented;
* major workflows documented;
* doctests added;
* lint enforcement introduced;
* documentation commands run;
* any known platform limitations.

Only after every public item passes `deny(missing_docs)`, rustdoc has no warnings, doctests pass as far as the environment supports, and the PR is open, finish with:

**Safe to rebase and merge.**
