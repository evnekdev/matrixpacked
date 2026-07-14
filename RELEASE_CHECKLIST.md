# First-release readiness audit

Audit date: 2026-07-14

Audited base: `master` at `6e534b96de230b46f920aee02e4baac30232c252`

Target release: `matrixpacked 0.1.0`

Statuses in this document are limited to: **passed**, **fixed in this PR**,
**deferred but non-blocking**, **blocking and requiring a separate PR**, and
**not testable on this host**.

## Release gate summary

- **passed** — The crate name and planned version are `matrixpacked 0.1.0`.
- **passed** — The crates.io crate endpoint returned HTTP 404, so neither the
  crate nor version is currently registered to another publisher. The name is
  available for the intended account to claim during the controlled first
  publication.
- **passed** — Neither the local repository nor GitHub has a `v0.1.0` tag.
- **passed** — GitHub reported no open pull requests, including no unmerged
  release PR that conflicts with this sequence.
- **passed** — The latest `master` CI run for the audited commit completed
  successfully: <https://github.com/evnekdev/matrixpacked/actions/runs/29306668646>.
- **blocking and requiring a separate PR** — The repository has no license
  file and the manifest has neither `license` nor `license-file`. Prompt 02
  must record the human license and copyright decision before publication.
- **blocking and requiring a separate PR** — The manifest does not declare
  `rust-version`. Prompt 02 must select and validate the MSRV.
- **blocking and requiring a separate PR** — `CHANGELOG.md` does not yet
  exist. Prompt 03 owns the release-candidate changelog.
- **deferred but non-blocking** — GitHub issue
  [#41](https://github.com/evnekdev/matrixpacked/issues/41) tracks making
  nalgebra triangular conversions backend-independent. The current native
  provider requirement is documented in `README.md` and
  `NALGEBRA_INTEROP.md`; the issue is intentionally not a `0.1.0` blocker.
- **deferred but non-blocking** — `EXAMPLE_COVERAGE.md` says there are 196
  top-level examples; the audited repository has 201 (173 `lapack_` and 28
  other examples). This documentation count should be refreshed in the
  release documentation work rather than expanding this audit PR.

The release sequence may proceed to Prompt 02. Publication remains blocked
until the license, attribution, and MSRV decisions are committed and all later
release gates pass.

## Repository and source audit

- **passed** — Inspected `Cargo.toml`, `Cargo.lock`, `README.md`, all source,
  examples, tests, workflows, and the user-facing packed-LAPACK, example,
  nalgebra, operations, and testing guides. The absent `CHANGELOG.md` and
  license files are recorded above.
- **passed** — Searches found no `todo!`, `unimplemented!`, `dbg!`, `println!`,
  or `eprintln!` in `src/`.
- **passed** — Searches found no handwritten `extern "C"` blocks or
  `link_name` attributes in `src/`; native declarations remain delegated to
  the `blas` and `lapack` crates.
- **passed** — Searches found no `FIXME`, `XXX`, `TEMPORARY`, or `HACK` markers
  in `src/`, `README.md`, or `docs/`.
- **passed** — Unsafe operations are confined to native backend calls behind
  safe APIs that validate dimensions, packed lengths, strides, workspace
  sizes, and right-hand-side layouts. Unit, oracle, property, edge-size, and
  failure-contract tests exercise these invariants.
- **passed** — Public API compilation succeeded with no features and with
  `nalgebra-interop`; `#![deny(missing_docs)]` and warning-denied docs passed.
- **passed** — Public exports were reviewed. No accidental debug, internal,
  or experimental API was identified, and no API redesign was made.
- **passed** — Empty default features are intentional. Native provider
  selection is delegated to the final application and documented accurately.
- **passed** — Only one bundled provider was activated at a time.

## Feature matrix

- **passed** — No features: library and all-target Rust compilation passed.
- **passed** — `nalgebra-interop`: all-target Rust compilation, Clippy, docs,
  and the provider-linked nalgebra test suite passed.
- **passed** — `intel-mkl-static`: all-target compilation, linked tests, and
  every example passed on Windows.
- **not testable on this host** — `openblas-static` dependency resolution was
  inspected, but its build script rejects non-vcpkg Windows builds with
  `Non-vcpkg builds are not supported on Windows`. This is a native-provider
  host limitation, not a Rust compilation failure. Linux CI uses documented
  system OpenBLAS instead.
- **passed** — `nalgebra-interop,intel-mkl-static`: all-target compilation and
  the full linked nalgebra suite passed.

## Manifest metadata inventory

- **passed** — `name = "matrixpacked"`.
- **passed** — `version = "0.1.0"`.
- **passed** — `edition = "2024"`.
- **blocking and requiring a separate PR** — `rust-version` is absent; owned
  by Prompt 02.
- **passed** — `description` is present and describes the packed structured
  matrix scope.
- **passed** — `repository` points to the public GitHub repository.
- **passed** — `documentation` points to docs.rs.
- **passed** — `readme = "README.md"` and the file is packaged.
- **blocking and requiring a separate PR** — `license`/`license-file` is
  absent; owned by the human decision in Prompt 02.
- **passed** — Five relevant keywords are present.
- **passed** — The `science` and `mathematics` categories are present.
- **fixed in this PR** — The existing `exclude` strategy now also excludes
  `/prompts_release/` and `/RELEASE_CHECKLIST.md` from crates.io packages.
- **passed** — An `include` allowlist is not used; explicit exclusions retain
  all source, examples, and linked user documentation.
- **passed** — `publish` is not restricted, which permits the later
  human-controlled crates.io publication step.
- **passed** — `0.1.0` is appropriate for the first release. Before `1.0.0`,
  incompatible API changes may occur in minor releases and must be documented.

## Package preview

- **fixed in this PR** — `cargo package --list` initially included
  `prompts_release/`; the manifest exclusion removes all Codex prompt folders
  and this internal audit checklist.
- **passed** — `.github/`, the root `cargo` artifact, existing prompt folders,
  and build output are excluded.
- **passed** — No ZIP, patch, IDE, credential, secret, or generated-doc
  artifact was present in the package preview.
- **passed** — All `src/` files, `README.md`, `docs/crate.md`, linked
  user-facing Markdown guides, tests, and the 201 documented examples are
  included.
- **blocking and requiring a separate PR** — No license file can be included
  until Prompt 02 records the human license decision.

## Dependencies

- **passed** — Normal runtime dependencies are limited to `blas`, `lapack`,
  `num-complex`, and `num-traits`, plus optional `nalgebra` or one optional
  provider when requested.
- **passed** — `nalgebra` is absent from the normal dependency tree unless
  `nalgebra-interop` is enabled.
- **passed** — `openblas-src` and `intel-mkl-src` are each absent unless their
  corresponding feature is enabled.
- **deferred but non-blocking** — Duplicate `getrandom`, `rand`,
  `rand_chacha`, and `rand_core` major versions occur only through development
  dependencies (`proptest`, `rand`, and `rand_chacha`), not the runtime tree.
  No dependency was upgraded merely for being newer.

## Validation record

Host toolchain: Windows MSVC, `rustc 1.97.0`, `cargo 1.97.0`.

- **passed** — `cargo fmt --all -- --check`.
- **passed** — `cargo check --lib --no-default-features`.
- **passed** — `cargo check --all-targets --no-default-features`.
- **passed** — `cargo check --all-targets --features nalgebra-interop`.
- **passed** — `cargo clippy --lib --no-default-features -- -D warnings`.
- **passed** — `cargo clippy --all-targets --features nalgebra-interop -- -D warnings`.
- **passed** — `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps --features nalgebra-interop`.
- **not testable on this host** — Provider-free
  `cargo test --doc --features nalgebra-interop` compiled and passed 18 of 19
  doctests, then one Hermitian doctest failed to link with 176 unresolved
  BLAS/LAPACK symbols. The same 19 doctests all passed with MKL. This was a
  native-provider/linker limitation, not a Rust compilation or assertion
  failure.
- **passed** — `cargo check --all-targets --features intel-mkl-static`.
- **passed** — `cargo test --features intel-mkl-static`: 51 library tests,
  4 inverse API tests, 195 oracle/property tests, and 18 doctests passed.
- **passed** — `cargo check --all-targets --features "nalgebra-interop,intel-mkl-static"`.
- **passed** — `cargo test --features "nalgebra-interop,intel-mkl-static"`:
  51 library, 4 inverse API, 15 validation, 7 full-triangular, 210 oracle,
  13 structured-conversion, 6 triangular-conversion, and 19 doctests passed.
- **passed** — The reproducible 96-case property tier passed all 11 matching
  tests with seed `557075679` (184 tests filtered out). Proptest emitted
  persistence-location warnings only; there were no Clippy findings or test
  failures.
- **passed** — The Windows non-LAPACK runner passed all 28 examples, including
  all 5 nalgebra examples, with MKL.
- **passed** — All 173 `lapack_` example `main` bodies passed in a single
  MKL-linked audit executable. This was used because per-executable Windows
  startup scanning made the repository's otherwise equivalent serial runner
  impractically slow. The temporary audit executable source was removed.
- **passed** — `cargo package --list`.
- **passed** — `cargo tree --duplicates`, `cargo tree -e normal`, and
  `cargo tree --features nalgebra-interop -e normal`.
- **not testable on this host** — The extended stress tier is Linux/OpenBLAS
  scheduled or manually dispatched CI. GitHub reported no runs yet for
  `extended.yml`, so there is no latest green extended result to cite.

## Final disposition

- **passed** — No Rust compilation failures remain in the supported feature
  matrix.
- **passed** — No Clippy findings, warning-denied rustdoc warnings, or test
  assertion failures remain.
- **fixed in this PR** — Release-only prompt and audit files no longer enter
  the package.
- **blocking and requiring a separate PR** — License, attribution, and MSRV
  decisions remain for Prompt 02; the changelog remains for Prompt 03.
- **passed** — No numerical functionality, tag, release, or publication was
  added or performed.
