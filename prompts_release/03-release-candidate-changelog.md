# Prompt 03 — Prepare the 0.1.0 release candidate

Work on https://github.com/evnekdev/matrixpacked

Start only after Prompt 02 is merged.

Create `agent/release-0.1.0-candidate`. Open a draft PR titled **Prepare matrixpacked 0.1.0**. Do not publish or create a tag. Finish only with:

**Safe to rebase and merge.**

## Goal

Prepare the reviewed source tree and human-facing release notes for `0.1.0`.

If `0.1.0` is already published or tagged, stop and ask which version to use.

## Version

Confirm:

```toml
version = "0.1.0"
```

Do not create a meaningless edit if already correct.

## Changelog

Create or update `CHANGELOG.md` with a clear structure:

```markdown
# Changelog

## [Unreleased]

## [0.1.0] - YYYY-MM-DD

### Added
...

### Known limitations
...
```

Use the planned publication date only if known. Summarize user-visible capabilities rather than every commit.

Cover:

- packed lower/upper matrices;
- symmetric, Hermitian, SPD/HPD matrices;
- owned/view/view-mut storage;
- arithmetic, multiplication, and rank updates;
- factorization, solve, inverse, conditions, refinement;
- simple/expert drivers;
- standard/generalized eigensolvers;
- reductions and format conversions;
- diagnostics;
- optional nalgebra interoperability;
- test and documentation quality.

Known limitations should include, where still true:

- linked numerical operations require a native BLAS/LAPACK provider;
- current Windows provider guidance;
- issue #41's provider dependency for triangular nalgebra conversions;
- pre-1.0 API stability expectations.

Do not describe issue #41 as a correctness defect.

## GitHub release notes

Create `RELEASE_NOTES_0.1.0.md` with:

1. overview;
2. key capabilities;
3. installation example;
4. backend examples;
5. optional nalgebra feature;
6. docs links;
7. known limitations;
8. MSRV;
9. first-public-release statement.

Avoid unsupported benchmark/performance claims.

## README release review

Review the README as rendered on crates.io. Confirm:

- clear purpose;
- supported structures;
- installation instructions;
- native provider requirement;
- short runnable example;
- optional nalgebra feature;
- docs/status/testing links;
- license;
- MSRV;
- pre-1.0 wording if desired.

Avoid repository-relative links that break in the packaged README.

Do not add crates.io/docs.rs badges before those pages exist unless they fail gracefully. Prompt 07 handles live badges.

## Checklist

Update `RELEASE_CHECKLIST.md` with release notes, changelog, README review, version confirmation, known limitations, and confirmation that no tag/publish occurred.

## Validation

```bash
cargo fmt --all -- --check
cargo check --all-targets --no-default-features
cargo check --all-targets --features nalgebra-interop
cargo test --doc --features nalgebra-interop
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop
cargo package --list
cargo package
git diff --check
```

Run provider-linked tests/examples. Inspect packaged README and changelog from the extracted `.crate`.

## Commit

```text
Prepare matrixpacked 0.1.0
```

## PR description

Summarize release documentation, version confirmation, known limitations, package verification, and commands run.
