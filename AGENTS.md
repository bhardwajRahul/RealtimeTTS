# RealtimeTTS release contract

- A release must use `tools/release_guard.py publish`; direct `twine upload` is not an accepted release path.
- Build wheel and sdist once from a clean commit. Install that exact wheel in the target runtime, run the focused runtime smoke, create an attestation with `tools/release_guard.py attest`, and publish those unchanged artifacts.
- The guard must inspect every linked worktree. If any worktree is dirty, stop and reconcile or explicitly preserve its changes; never discard them to make a release pass.
- Do not copy edited Python files directly into `site-packages`. Emergency fixes still go through a locally built wheel and an attestation before deployment is considered complete.
- Run the relevant test suite once before building. The parity guard is intentionally hash-based and fast; do not repeat long suites merely to satisfy it.
