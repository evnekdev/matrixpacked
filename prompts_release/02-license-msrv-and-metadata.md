# Prompt 02 — Finalize license, MSRV, and package metadata

Work on https://github.com/evnekdev/matrixpacked

Start only after Prompt 01 is merged and `RELEASE_CHECKLIST.md` permits continuation.

## Workflow

Create `agent/release-license-metadata` from latest `master`. Open a draft PR titled **Finalize release metadata**. Do not publish or change the crate version. Finish only with:

**Safe to rebase and merge.**

## Goal

Resolve all legal and manifest metadata needed for a responsible first crates.io publication.

## Human license decision gate

Inspect:

```text
LICENSE
LICENSE-APACHE
LICENSE-MIT
COPYING*
NOTICE*
Cargo.toml
README.md
```

If no explicit license decision exists, stop and ask the user to choose.

A common recommendation for Rust libraries is:

```text
MIT OR Apache-2.0
```

But do not assume it without explicit confirmation.

Never invent:

- copyright holder;
- legal name;
- organization;
- year;
- license exception.

After confirmation, add exact standard license files. For dual licensing, conventionally add `LICENSE-MIT` and `LICENSE-APACHE`, and set:

```toml
license = "MIT OR Apache-2.0"
```

Use exact SPDX syntax. Avoid setting both `license` and `license-file` without a specific reason.

## Manifest metadata

Audit and finalize:

```toml
[package]
name = "matrixpacked"
version = "0.1.0"
edition = "2024"
rust-version = "..."
description = "..."
repository = "https://github.com/evnekdev/matrixpacked"
documentation = "https://docs.rs/matrixpacked"
readme = "README.md"
license = "..."
keywords = [...]
categories = [...]
publish = ["crates-io"]
```

### Description

Keep it concise and specific. Mention packed structured matrices and BLAS/LAPACK integration without unsupported performance claims.

### Keywords and categories

Use no more than five valid crates.io keywords. Verify categories exist. Keep current values unless a clear improvement is justified.

### Registry restriction

Add:

```toml
publish = ["crates-io"]
```

unless another registry is intentionally supported.

## Determine MSRV

Do not guess `rust-version`.

Because the crate uses edition 2024, begin with the minimum toolchain supporting that edition, then account for dependencies and language features.

At minimum:

```bash
rustup toolchain install <candidate>
cargo +<candidate> check --lib --no-default-features
cargo +<candidate> check --lib --features nalgebra-interop
```

Test linked functionality where practical. Distinguish core-crate MSRV from optional provider dependency constraints, but prefer one documented MSRV that supports intended normal use.

Set only the verified result:

```toml
rust-version = "X.Y"
```

Record commands and results in `RELEASE_CHECKLIST.md`.

## README and crate docs

Add a License section linked to actual license files. Document the verified MSRV without promising an indefinite policy.

## Secrets audit

Search tracked files:

```bash
rg -n -i 'api[_-]?token|cargo[_-]?token|crates\.io.*token|secret|password' .
```

Evaluate false positives. Never commit Cargo credentials, tokens, environment dumps, or shell history.

## Package verification

Run:

```bash
cargo metadata --no-deps --format-version 1
cargo package --list
cargo package
```

Do not use `--no-metadata`, `--allow-dirty`, or `--no-verify`.

Extract the generated `.crate` and confirm README and license files are included.

## Validation

```bash
cargo fmt --all -- --check
cargo check --all-targets --no-default-features
cargo check --all-targets --features nalgebra-interop
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo package
git diff --check
```

Run linked tests with the supported provider.

## Checklist update

Record license choice, attribution source, files added, SPDX expression, verified MSRV, final metadata, and package result.

## Commit

```text
Finalize release metadata
```

## PR description

Include the explicit user-approved license choice, attribution source, MSRV evidence, metadata changes, and package verification.
