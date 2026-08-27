# Releasing RealtimeTTS

RealtimeTTS releases promote the exact artifact already exercised by the target runtime. The release guard adds only hashing and Git-status checks; it does not repeat the full test suite.

1. Reconcile every intended change into one clean commit. All linked worktrees must be clean.
2. Run the relevant tests once, then build wheel and sdist once:

   ```powershell
   python -m build --sdist --wheel
   python -m twine check --strict dist/*
   ```

3. Install the newly built wheel in the target runtime and run its focused server/engine smoke test. Do not copy individual `.py` files.
4. From that runtime's Python, attest the installed package against the unchanged wheel and sdist:

   ```text
   python tools/release_guard.py attest \
     --repo <clean-source-root> \
     --component RealtimeTTS \
     --distribution realtimetts \
     --package-dir RealtimeTTS \
     --wheel <wheel> \
     --artifact <sdist> \
     --runtime-package-dir <runtime-site-packages>/RealtimeTTS \
     --dependency realtimetts-qwen-native \
     --runtime-label <service-or-venv> \
     --output <deployment-manifest.json>
   ```

5. Publish only through the guard, using the exact same local artifacts and the returned deployment manifest:

   ```text
   python tools/release_guard.py publish \
     --repo <clean-source-root> \
     --package-dir RealtimeTTS \
     --wheel <wheel> \
     --artifact <sdist> \
     --manifest <deployment-manifest.json>
   ```

The command fails before Twine runs if a linked worktree is dirty, the source commit changed, package files differ, or either artifact hash differs from what was deployed.
