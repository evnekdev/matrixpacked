Work on the public GitHub repository:

https://github.com/evnekdev/matrixpacked

Create a fresh branch from the latest `master`:

```text
agent/eliminate-build-warnings
```

Commit and PR title:

```text
Eliminate build warnings
```

# Goal

Make the crate compile warning-free across all supported targets, features, examples, tests, and documentation by fixing the underlying causes.

Do not silence warnings globally.

Do not add broad attributes such as:

```rust
#![allow(warnings)]
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(clippy::all)]
```

Do not add file-level or module-level `allow` attributes merely to make output quiet.

A narrowly scoped `#[allow(...)]` is acceptable only when:

1. the warning is unavoidable;
2. the code is intentional;
3. the reason is documented directly beside the attribute;
4. no reasonable code change can remove the warning.

# Workflow

1. Fetch the repository.
2. Switch to `master`.
3. Pull with fast-forward only.
4. Confirm the working tree is clean.
5. Create:

```text
agent/eliminate-build-warnings
```

6. Do not modify `master` directly.
7. Keep this PR focused on warning fixes.
8. Do not perform broad API redesigns.
9. Do not add documentation-completeness enforcement yet; that belongs to the following PRs.

# Phase 1: warning inventory

Run every command below and save the warnings in a temporary checklist before changing code.

## Default configuration

```bash
cargo clean
cargo check --all-targets
cargo test --no-run
cargo check --examples
cargo doc --no-deps
```

## Feature configurations

Run each supported backend independently:

```bash
cargo check --all-targets --features openblas-static
cargo check --examples --features openblas-static
cargo test --no-run --features openblas-static
cargo doc --no-deps --features openblas-static
```

On a platform that supports MKL:

```bash
cargo check --all-targets --features intel-mkl-static
cargo check --examples --features intel-mkl-static
cargo test --no-run --features intel-mkl-static
cargo doc --no-deps --features intel-mkl-static
```

Also verify the feature-free library:

```bash
cargo check --lib --no-default-features
```

Do not enable `openblas-static` and `intel-mkl-static` simultaneously unless the crate explicitly supports that combination.

## Clippy

Run:

```bash
cargo clippy --all-targets --no-default-features -- -D warnings
cargo clippy --all-targets --features openblas-static -- -D warnings
```

Run the MKL equivalent on supported Windows environments.

## Rustdoc

Run:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

Use the Windows equivalent when necessary:

```bat
set RUSTDOCFLAGS=-D warnings
cargo doc --no-deps
```

Do not enable `missing_docs` yet unless it is already enabled. This PR is for existing compiler, Clippy, and rustdoc warnings.

# Phase 2: classify every warning

Classify warnings by cause:

* unused imports;
* unused variables;
* unnecessary `mut`;
* dead private code;
* unreachable code;
* redundant qualification;
* needless borrowing;
* manual range or iterator logic;
* large or complex function signatures;
* naming problems;
* unsafe-operation warnings;
* deprecated APIs;
* unexpected `cfg` conditions;
* feature-specific imports;
* example-only warnings;
* test-only warnings;
* rustdoc broken links;
* invalid HTML or Markdown;
* unused results;
* suspicious casts;
* numeric conversion risks.

Do not fix warnings mechanically without checking semantics.

# Required fixing principles

## Unused imports

For feature-dependent imports, use precise `#[cfg(...)]` placement.

Do not keep unconditional imports and silence them.

## Unused variables

If the value is genuinely unnecessary, remove it.

If it is required for ABI shape or pattern matching, use a meaningful underscore-prefixed name and document why where non-obvious.

Do not rename all variables to `_x` without understanding their purpose.

## Dead private code

Determine whether the item is:

* obsolete and should be deleted;
* intended for an upcoming feature and should remain on a feature branch rather than `master`;
* accidentally unused due to a missing integration;
* useful only in tests and should move behind `#[cfg(test)]`.

Delete obsolete private code.

Do not remove public APIs merely because internal compilation reports them as unused.

## Clippy suggestions

Accept Clippy suggestions only when they:

* preserve behavior;
* do not obscure LAPACK argument ordering;
* do not make unsafe numerical code harder to audit;
* do not add allocations;
* do not weaken checked conversions.

For FFI dispatch code, clarity and exact correspondence to LAPACK signatures are more important than shortening code.

## Numeric casts

Replace unchecked narrowing conversions with existing checked helpers where appropriate.

Do not use `as i32` for dimensions when the crate already has checked conversion utilities.

## Unsafe code

Do not add large `unsafe` blocks to avoid warnings.

Keep unsafe scopes minimal.

If Rust 2024 emits `unsafe_op_in_unsafe_fn`, add explicit narrowly scoped `unsafe { ... }` blocks rather than suppressing the lint.

## Feature providers

Inspect:

```rust
#[cfg(feature = "openblas-static")]
use openblas_src as _;

#[cfg(feature = "intel-mkl-static")]
use intel_mkl_src as _;
```

Preserve provider-linking behavior.

Do not remove intentionally underscore-imported provider crates as “unused”.

## Examples and tests

Every example under `examples/` must compile without warnings.

Test helper modules must also be warning-free.

Do not exclude examples from validation merely because there are many.

# Optional lint policy

After all warnings are fixed, add a moderate project lint policy only if it remains maintainable.

Possible crate-level settings:

```rust
#![warn(rust_2018_idioms)]
#![warn(unused_lifetimes)]
#![warn(unused_qualifications)]
```

Because this crate uses edition 2024, verify whether each lint is still useful.

Do not add extremely opinionated Clippy lint groups such as:

```rust
#![warn(clippy::pedantic)]
#![warn(clippy::restriction)]
```

in this PR.

CI can run `clippy -D warnings` without embedding all Clippy policy into the library source.

# Validation

At the end, rerun all applicable commands from the inventory.

Required minimum:

```bash
cargo fmt --all
cargo fmt --all -- --check
cargo check --all-targets
cargo check --examples
cargo test --no-run
cargo clippy --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
git diff --check
```

Run backend-feature variants supported by the current machine.

If a native backend cannot be built on that platform, clearly distinguish:

* provider installation failure;
* linker failure;
* Rust warning;
* Rust compilation error.

Do not claim a configuration passed if it could not be tested.

# Scope review

Inspect:

```bash
git diff master...HEAD --stat
git diff master...HEAD
```

Confirm every changed line is related to warning cleanup or narrowly necessary refactoring.

The PR description must contain:

* commands run;
* warning categories found;
* root-cause fixes made;
* any narrow `allow` attributes retained and why;
* feature/platform combinations tested;
* configurations not tested and why.

Open a draft PR targeting `master`.

Do not declare completion while warnings remain in any tested configuration.

Only after the final diff is inspected, no additional commits are planned, and the PR is open, finish with exactly:

**Safe to rebase and merge.**
