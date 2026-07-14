# Prompt 06 — Human-controlled publication of matrixpacked 0.1.0

Work on https://github.com/evnekdev/matrixpacked

Start only after Prompt 05 is merged.

This step does not normally create a branch or PR.

## Goal

Verify the exact merged release state and perform the irreversible crates.io upload only after explicit current-thread authorization.

Never ask the user to paste a crates.io token.

## Phase A — Verify merged release state

1. Fetch remote changes.
2. Switch to `master`.
3. Pull with fast-forward only.
4. Confirm clean tree.
5. Read `RELEASE_CHECKLIST.md`, `PUBLISHING.md`, `CHANGELOG.md`, and `RELEASE_NOTES_0.1.0.md`.
6. Confirm version `0.1.0`, green CI, no tag, and no existing publication.

Run:

```bash
cargo publish --dry-run
```

Do not use `--allow-dirty` or `--no-verify`.

Report the exact final merged commit SHA and dry-run result.

## Phase B — Explicit approval gate

Stop and require a current explicit instruction equivalent to:

```text
Publish matrixpacked 0.1.0 from commit <sha> to crates.io now.
```

Do not treat earlier roadmap approval as authorization.

## Phase C — Authentication

Use only local Cargo authentication configured by the user, such as `cargo login` or a secure credential provider/environment configuration.

Never print token values, credentials files, or authentication headers.

If authentication is unavailable, instruct the user to configure it locally and stop.

## Phase D — Publish

After explicit approval and confirmed local authentication, run exactly:

```bash
cargo publish
```

Do not add `--allow-dirty`, `--no-verify`, or another registry.

If Cargo uploads and then times out polling the index:

1. do not immediately retry;
2. check crates.io for `matrixpacked 0.1.0`;
3. retry only if the registry clearly confirms no acceptance.

## Phase E — Immediate verification

Confirm:

- version appears on crates.io;
- metadata, license, README, links, owners, dependencies, and features are correct;
- downloadable package contents match expectations.

Do not tag if success is uncertain.

## Failure handling

- Pre-upload failure: fix in a focused PR; do not publish.
- Registry rejection before acceptance: correct it and verify whether the version remains available.
- Accepted defective release: do not replace it; prepare `0.1.1` and consider yanking only for a serious defect.

## Completion response

Report published crate/version, source commit, registry verification, docs.rs status if available, and readiness for Prompt 07.

Do not use the merge-safe phrase because this step has no normal PR.
