# Prompt 06 — Document and enforce nalgebra interoperability quality

Work on:

https://github.com/evnekdev/matrixpacked

## Starting conditions

Start after Prompt 05 is merged.

Create:

```text
agent/nalgebra-interop-documentation
```

## Goal

Complete user documentation, examples, package metadata, and CI coverage for the optional nalgebra interoperability feature.

This task should make the feature ready for public use and prevent regressions.

## Public API documentation

Document every public interoperability item:

- feature flag;
- methods on `FullTriangular`;
- methods on `PackedLower`;
- methods on `PackedUpper`;
- methods on `PackedSymmetric`;
- methods on `PackedHermitian`;
- methods on `PackedSPD`;
- tolerance type;
- extraction methods;
- strict validated constructors;
- relevant errors.

For every conversion method document:

- whether it allocates;
- whether it clones;
- whether it consumes;
- whether it validates structure;
- whether it ignores one triangle;
- output storage order;
- error conditions;
- scalar restrictions;
- backend requirements.

## Crate-level guide

Add a dedicated section to the crate documentation:

```text
Nalgebra interoperability
```

Explain how to enable:

```toml
matrixpacked = {
    version = "...",
    features = ["nalgebra-interop"]
}
```

Include examples.

### Example 1: packed lower to nalgebra

```rust
let packed = PackedLower::from_vec(...)?;
let dense = packed.to_dmatrix()?;
```

Adapt return types to the implemented API.

### Example 2: strict symmetric conversion

```rust
let packed = PackedSymmetric::try_from_dmatrix(
    &dense,
    ConversionTolerance::new(...),
)?;
```

### Example 3: explicit triangle extraction

Show that extraction intentionally ignores the opposite triangle.

### Example 4: SPD validation

Show nalgebra Cholesky-backed validation.

## Important conceptual documentation

Explain:

### Conversion, not viewing

A nalgebra matrix cannot directly view traditional packed storage because its rectangular stride model does not match variable-length packed columns.

All conversions allocate unless an owned full `n × n` buffer can be moved directly.

### Full storage cost

Packed storage:

```text
n(n+1)/2
```

Nalgebra full matrix:

```text
n²
```

Document memory growth.

### Complex symmetric versus Hermitian

Give a concrete example using `2 + 3i`.

### Extraction versus validation

Make the naming distinction prominent.

### LAPACK versus pure Rust paths

Document:

- LAPACK format conversions use specialized LAPACK routines;
- logical structured expansion into nalgebra uses pure Rust where implemented;
- why this distinction exists.

Do not overpromise backend independence if triangular paths still call LAPACK.

## README

Add a concise optional-integration section.

Do not make nalgebra appear mandatory.

Include:

```text
cargo add matrixpacked --features nalgebra-interop
```

only if that syntax fits the intended release documentation.

## Examples

Review all nalgebra examples.

Ensure:

- each is feature-gated with `required-features`;
- each compiles;
- examples use current API names;
- examples are included in relevant run scripts only when the feature is enabled, or documented separately.

Do not make the ordinary LAPACK example runner fail because optional nalgebra examples require a feature.

Consider a separate script:

```text
scripts_run_nalgebra_examples.sh
scripts_run_nalgebra_examples.bat
```

It should run examples with:

```text
--features nalgebra-interop
```

and any required backend feature.

Only add these scripts if they improve usability.

## Cargo metadata

Ensure the optional feature appears clearly in generated docs.

If docs.rs metadata exists, decide whether to enable:

```text
nalgebra-interop
```

for docs.rs.

Preferred:

```toml
[package.metadata.docs.rs]
features = ["nalgebra-interop"]
```

only if documentation can build reliably without a native provider.

Do not enable a problematic static BLAS backend on docs.rs.

## CI

Update CI to run:

```bash
cargo check --all-targets --features nalgebra-interop
cargo test --features nalgebra-interop
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
```

Also retain feature-free checks:

```bash
cargo check --all-targets --no-default-features
cargo test --no-default-features
```

This proves optional isolation.

Do not duplicate expensive jobs unnecessarily.

## Missing-doc enforcement

If the documentation PR series has already enabled:

```rust
#![deny(missing_docs)]
```

ensure all interoperability APIs comply.

If documentation enforcement has not yet been introduced, do not preempt the broader documentation prompt unless this feature can cleanly document all its own public items.

At minimum, no new missing-doc warnings should be introduced.

## Validation

Run:

```bash
cargo fmt --all -- --check
cargo check --all-targets --no-default-features
cargo test --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo test --features nalgebra-interop
cargo test --doc --features nalgebra-interop
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo package --list
git diff --check
```

Manually inspect generated Rustdoc.

Verify all links and examples.

## Commit and PR

Commit:

```text
Document nalgebra interoperability
```

Branch:

```text
agent/nalgebra-interop-documentation
```

PR title:

```text
Document nalgebra interoperability
```

The PR description must summarize:

- user guide;
- examples;
- feature gating;
- docs.rs handling;
- CI jobs;
- backend requirements;
- extraction and validation distinctions.

Only after CI is green, documentation is manually reviewed, no more commits are planned, and the PR is open, finish with:

**Safe to rebase and merge.**
