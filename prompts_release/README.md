# matrixpacked first crates.io release prompt series

Execute these prompts strictly in order. Merge each pull request before starting the next numbered prompt.

1. `01-release-readiness-audit.md`
2. `02-license-msrv-and-metadata.md`
3. `03-release-candidate-changelog.md`
4. `04-package-and-docsrs-verification.md`
5. `05-publish-dry-run-and-final-gate.md`
6. `06-human-controlled-publication.md`
7. `07-post-publication-release-and-verification.md`

## Important release rules

- Publishing a version to crates.io is irreversible. A published version cannot be replaced.
- Never share a crates.io API token in a prompt, issue, commit, PR, log, or chat.
- Codex must not run `cargo publish` until the user explicitly authorizes that exact upload in Prompt 06.
- Every release operation must use a clean checkout of the exact reviewed commit.
- The planned first version is `0.1.0`, unless that version is already published when the prompts are run.
- GitHub issue #41 is not a release blocker. Keep its current limitation documented.
- Do not add new numerical features during release preparation.
- If a release-blocking correctness or packaging defect is found, stop the sequence and fix it in a separate focused PR.

## New Codex thread instruction

Tell Codex:

> Read `prompts_release/README.md`, then execute only the first unfinished numbered prompt. Complete its PR fully and stop after the exact merge-safe message. Do not begin the next prompt until I confirm the preceding PR has been merged.
