Work on:

https://github.com/evnekdev/matrixpacked

Start after the comprehensive documentation PR has been merged.

Create:

```text
agent/enforce-quality-gates
```

Commit and PR title:

```text
Enforce warning and documentation quality
```

# Goal

Add durable CI and package-quality gates so future changes cannot reintroduce warnings, undocumented public APIs, broken Rustdoc links, failing doctests, or packaging defects.

# Audit existing workflows

Inspect:

```text
.github/workflows/
Cargo.toml
src/lib.rs
README.md
```

Do not create duplicate workflows if an existing CI workflow can be extended cleanly.

# Required CI jobs

## 1. Formatting

```bash
cargo fmt --all -- --check
```

## 2. Compiler warnings

Run:

```bash
RUSTFLAGS="-D warnings" cargo check --all-targets
```

Be careful: dependency warnings should not normally be promoted through `--cap-lints` behavior, but verify the actual command.

Test the default feature configuration and the primary supported backend configuration separately.

## 3. Clippy

```bash
cargo clippy --all-targets -- -D warnings
```

Run with the primary backend feature where required to compile examples.

## 4. Unit and integration tests

Run:

```bash
cargo test
```

with the backend feature needed for native operations.

## 5. Doctests

Run:

```bash
cargo test --doc
```

Ensure native provider setup is available.

## 6. Rustdoc

Run:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

Because `src/lib.rs` should now contain:

```rust
#![deny(missing_docs)]
```

this also enforces public API completeness.

Explicitly ensure broken intra-doc links fail the build.

## 7. Examples

Compile every example:

```bash
cargo check --examples
```

Run the repository’s example runner where CI runtime remains reasonable.

If running every example is too expensive on each push, separate:

* compile every example on every PR;
* run all examples on scheduled/manual CI.

## 8. Package verification

Run:

```bash
cargo package --allow-dirty
```

Prefer a clean checkout so `--allow-dirty` is unnecessary.

Inspect the package file list:

```bash
cargo package --list
```

Ensure tests, docs, licenses, README, and required source files are included, while generated files and local artifacts are excluded.

# Cargo metadata

Complete `[package]` metadata after verifying exact values:

```toml
description = "..."
repository = "https://github.com/evnekdev/matrixpacked"
documentation = "https://docs.rs/matrixpacked"
readme = "README.md"
license = "MIT OR Apache-2.0"
keywords = ["matrix", "lapack", "blas", "packed", "linear-algebra"]
categories = ["mathematics", "science"]
```

Use only licenses actually present in the repository.

Do not invent a homepage.

Crates.io allows at most five keywords; verify all metadata constraints.

Add `rust-version` only after determining the actual MSRV through testing.

Because the crate uses edition 2024, select an MSRV compatible with that edition and all dependencies.

# docs.rs metadata

Add docs.rs configuration only if needed.

Possible form:

```toml
[package.metadata.docs.rs]
features = ["..."]
rustdoc-args = ["--cfg", "docsrs"]
```

Do not enable a native static backend on docs.rs if it makes documentation builds fail.

The library’s documentation should compile without requiring final native linking whenever possible.

Use:

```rust
#![cfg_attr(docsrs, feature(doc_cfg))]
```

only if nightly docs.rs functionality is genuinely used and compatible.

Do not add unnecessary nightly-only code.

# Feature documentation

If provider features are mutually exclusive, add a compile-time error or clearly documented policy only if simultaneous activation is genuinely invalid.

Do not introduce this behavior casually; verify how downstream feature unification should work.

# README badges

Add only useful badges:

* CI;
* crates.io after publication;
* docs.rs after publication;
* license.

Do not add dead badges before URLs exist.

# CI platform strategy

At minimum test Linux with a reliable BLAS/LAPACK provider.

Add Windows with MKL if the setup is stable.

Do not add a Windows OpenBLAS configuration known to fail.

Prefer a small, reliable matrix over a large flaky one.

# Required validation

Before opening the PR, run locally as many CI commands as the environment supports.

Inspect the workflow YAML carefully for:

* valid action versions;
* caching correctness;
* native dependencies;
* environment-variable syntax;
* feature selection;
* shell differences on Windows.

The PR description must include:

* jobs added;
* platforms and backends tested;
* package metadata added;
* documentation enforcement mechanism;
* commands run locally;
* any CI-only validation.

Open a draft PR.

Only after CI is green, the final diff is reviewed, and no more commits are planned, finish with:

**Safe to rebase and merge.**
