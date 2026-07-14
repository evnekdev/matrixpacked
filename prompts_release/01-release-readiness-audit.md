# Prompt 01 — Perform a first-release readiness audit

Work on https://github.com/evnekdev/matrixpacked

## Workflow

1. Fetch all remote changes.
2. Switch to `master` and pull with fast-forward only.
3. Confirm the working tree is clean.
4. Create `agent/release-readiness-audit` from current `master`.
5. Do not modify `master` directly.
6. Open a draft PR titled **Audit first release readiness**.
7. Do not add new numerical functionality.
8. Finish only after the PR is complete, with:

**Safe to rebase and merge.**

## Goal

Perform a complete release audit and create `RELEASE_CHECKLIST.md`. Commit only small, unambiguous, non-behavioral release-readiness fixes.

For every item, record one status:

- passed;
- fixed in this PR;
- deferred but non-blocking;
- blocking and requiring a separate PR;
- not testable on this host.

## Repository and registry state

Inspect at least:

```text
Cargo.toml
Cargo.lock
README.md
CHANGELOG.md
LICENSE*
src/
examples/
tests/
.github/workflows/
PACKED_LAPACK_FUNCTIONS.md
EXAMPLE_COVERAGE.md
NALGEBRA_INTEROP.md
TESTING.md
```

Confirm:

- crate name is `matrixpacked`;
- planned version is `0.1.0`;
- `matrixpacked 0.1.0` is not already published;
- the crates.io name is available to the intended account;
- no `v0.1.0` release tag already exists;
- no unmerged release PR conflicts with this sequence;
- issue #41 remains documented and non-blocking.

If the crate name belongs to another publisher, stop and report a blocker. Do not publish a placeholder crate.

## Source and API audit

Search and evaluate all hits:

```bash
rg 'todo!\(|unimplemented!\(|dbg!\(|println!\(|eprintln!\(' src
rg 'extern\s+"C"|link_name' src
rg 'FIXME|XXX|TEMPORARY|HACK' src README.md docs
```

Confirm:

- no production `todo!()` or `unimplemented!()` remains;
- no accidental debug output remains;
- no handwritten BLAS/LAPACK FFI was reintroduced;
- unsafe blocks have clear invariants and test coverage;
- public APIs compile with and without `nalgebra-interop`;
- provider-feature documentation is accurate;
- no accidental experimental public API is exposed.

Do not redesign the API in this audit PR. Open a focused issue for a substantial problem.

## Formatting, warnings, Clippy, and docs

Run provider-free checks:

```bash
cargo fmt --all -- --check
cargo check --lib --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo clippy --lib --no-default-features -- -D warnings
cargo clippy --all-targets --features nalgebra-interop -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo test --doc --features nalgebra-interop
```

Run linked checks with the host-supported provider.

Windows:

```bash
cargo check --all-targets --features intel-mkl-static
cargo test --features intel-mkl-static
cargo check --all-targets --features "nalgebra-interop,intel-mkl-static"
cargo test --features "nalgebra-interop,intel-mkl-static"
```

Linux: use the repository's documented OpenBLAS setup.

Distinguish native-provider/linker failures from Rust compilation failures.

## Test tiers

Run:

- ordinary tests;
- nalgebra oracle tests;
- normal property-test tier;
- all examples or the repository example runners;
- nalgebra examples;
- doctests.

The extended stress tier may rely on the latest green scheduled CI. Record its latest known status.

Record test counts where practical.

## Feature matrix

Audit:

```text
no features
nalgebra-interop
openblas-static
intel-mkl-static
nalgebra-interop + one provider
```

Do not activate both provider features unless the project explicitly supports it. Confirm empty default features are intentional.

## Manifest metadata inventory

Record the status of:

```text
name
version
edition
rust-version
description
repository
documentation
readme
license or license-file
keywords
categories
exclude/include
publish
```

Do not select or add a license unless the repository already contains an unambiguous license decision. Missing licensing is a Prompt 02 blocker.

## Package contents preview

Run:

```bash
cargo package --list
```

Confirm exclusion of:

- `.github/`;
- Codex prompt folders;
- ZIP and patch artifacts;
- generated docs/build artifacts;
- IDE files;
- credentials and secrets.

Confirm inclusion of:

- all source files;
- README;
- user-facing documentation needed by packaged links;
- examples referenced in docs;
- license files once present.

Do not use `--allow-dirty`.

## Dependency audit

Run:

```bash
cargo tree --duplicates
cargo tree -e normal
cargo tree --features nalgebra-interop -e normal
```

Review duplicate major versions, unexpected runtime dependencies, nalgebra optionality, and provider optionality. Do not upgrade dependencies merely because newer versions exist.

## SemVer and CI

Confirm `0.1.0` is appropriate and document pre-1.0 stability expectations.

Inspect latest `master` CI. Required jobs must be green. Record scheduled/manual jobs that have not run recently.

## Allowed changes

Allowed:

- `RELEASE_CHECKLIST.md`;
- stale-link corrections;
- package exclusion fixes;
- removal of accidental generated files;
- small README accuracy fixes.

Not allowed:

- license selection;
- version changes;
- release changelog section;
- tags;
- publication;
- API redesign;
- issue #41 implementation.

## Final validation

```bash
cargo fmt --all -- --check
cargo check --all-targets --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo package --list
git diff --check
git diff master...HEAD --stat
git diff master...HEAD
```

Run linked tests with the supported provider.

## Commit

```text
Audit first release readiness
```

## PR description

Summarize blockers, non-blocking limitations, commands run, backend/test coverage, package file-list findings, metadata gaps, and whether the release sequence may proceed.
