# Prompt 04 — Verify the exact package and docs.rs readiness

Work on https://github.com/evnekdev/matrixpacked

Start only after Prompt 03 is merged.

Create `agent/verify-0.1.0-package`. Open a draft PR titled **Verify matrixpacked 0.1.0 package**. Finish only with:

**Safe to rebase and merge.**

## Goal

Test the exact artifact Cargo will publish, from a clean checkout, and verify docs.rs-like documentation conditions.

Avoid source changes unless a packaging defect is found.

## Clean checkout

Run:

```bash
git status --short
```

It must be empty. Do not use `--allow-dirty`, `--no-verify`, or `--no-metadata`.

## Build and inspect the package

```bash
cargo package --list
cargo package -v
```

Record the exact archive name and calculate a checksum.

Extract the `.crate` into a temporary directory outside the working tree. Inspect:

- normalized `Cargo.toml`;
- `Cargo.toml.orig`;
- `.cargo_vcs_info.json`;
- source;
- README;
- changelog;
- license files;
- examples;
- packaged user documentation;
- absence of excluded prompts, CI files, ZIPs, patches, build artifacts, and credentials.

`.cargo_vcs_info.json` should describe a clean source state.

Do not commit extracted contents.

## Test the extracted artifact

Inside the extracted package:

```bash
cargo check --lib --no-default-features
cargo check --lib --features nalgebra-interop
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo test --doc --features nalgebra-interop
```

Then run linked tests with the supported provider.

Windows:

```bash
cargo test --features intel-mkl-static
cargo test --features "nalgebra-interop,intel-mkl-static"
```

Linux: use the repository's documented OpenBLAS setup.

Verify the package independently of workspace context.

## Manifest integrity

Confirm:

- no normal path dependencies;
- all normal dependencies have registry-compatible versions;
- optional dependencies remain optional;
- nalgebra is absent when `nalgebra-interop` is disabled;
- provider crates remain optional;
- package metadata is preserved correctly in normalized Cargo.toml.

## docs.rs readiness

Inspect `[package.metadata.docs.rs]`.

The intended feature set should include `nalgebra-interop` if it builds without a provider. Do not enable OpenBLAS or MKL for docs.rs unless unavoidable.

Verify:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
```

Check:

- crate landing page;
- all public APIs;
- feature-gated nalgebra APIs;
- no broken links or missing docs;
- no native link required during doc generation.

Do not add nightly-only docs features merely for cosmetic annotations.

## Packaged README review

Check for broken images/links, excluded-file references, wrong version/feature names, unreleased placeholders, and incorrect backend guidance.

## Example compilation

Compile every packaged example with its required features. Separate provider-free, provider-linked, and nalgebra examples.

## External consumer projects

Create temporary projects outside the repo.

### Core consumer

Use a path to the extracted package and compile a storage-only program.

### Nalgebra consumer

Enable `nalgebra-interop` and compile a conversion program.

### Numerical consumer

Enable the host provider and run one solve.

These checks catch missing files and accidental workspace dependence.

## Checklist update

Record package archive name, checksum, included-file review, extracted checks, docs.rs-like result, consumer tests, and blockers.

If a small package defect is fixed, repeat the entire package verification. Move correctness/API defects to separate PRs.

## Validation

```bash
cargo package --list
cargo package -v
git diff --check
git diff master...HEAD
```

## Commit

```text
Verify matrixpacked 0.1.0 package
```

## PR description

Include archive verification, contents, extracted-build commands, docs.rs readiness, external consumer results, and fixes.
