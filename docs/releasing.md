# Releasing RealtimeTTS

RealtimeTTS releases promote the exact artifact already exercised by the target runtime. The release guard adds only hashing and Git-status checks; it does not repeat the full test suite.

1. Reconcile every intended change into one clean commit. All linked worktrees must be clean.
2. Run the relevant tests once, then build wheel and sdist once:

   ```powershell
   python -m build --sdist --wheel
   python -m twine check --strict dist/*
   ```

3. Install the newly built wheel in the target runtime. Prove the service's current `ExecStart` uses that Python, restart to a new healthy PID, and run the focused server/engine smoke test, including all seven configured language routes. Do not copy individual `.py` files.
4. As the final runtime step, attest the installed package against the unchanged wheel and sdist. The Linux private key stays at `/home/lon/.config/codex-release-guard/signing_ed25519`; transfer only the manifest and detached `.sig` afterward:

   ```text
   python tools/release_guard.py attest \
     --repo <clean-source-root> \
     --component RealtimeTTS \
     --distribution realtimetts \
     --package-dir RealtimeTTS \
     --wheel <wheel> \
     --sdist <sdist> \
     --runtime-package-dir <runtime-site-packages>/RealtimeTTS \
     --runtime-module RealtimeTTS \
     --runtime-python <exact-service-venv>/bin/python \
     --dependency realtimetts-qwen-native \
     --runtime-label <service-or-venv> \
     --signer linux-services \
     --signing-key-file /home/lon/.config/codex-release-guard/signing_ed25519 \
     --output <deployment-manifest.json>
   ```

5. Within 30 minutes, publish only through the guard using the exact same local artifacts, manifest, and `.sig`. When publishing from Windows against the fresh Linux evidence, use the trusted public-key list and the explicit remote-attestation mode:

   ```text
   python tools/release_guard.py publish \
     --repo <clean-source-root> \
     --package-dir RealtimeTTS \
     --wheel <wheel> \
     --sdist <sdist> \
     --manifest <deployment-manifest.json> \
     --signature-file <deployment-manifest.json.sig> \
     --allowed-signers-file C:/Users/Start/.codex/release-state/allowed_signers \
     --signer linux-services \
     --allow-remote-attestation \
     --repository pypi \
     --remote origin \
     --branch master \
     --tag v<VERSION>
   ```

The command fails before Twine runs if a linked worktree is dirty, the signed Linux evidence is invalid or older than 30 minutes, the source commit or package identity changed, source/wheel/sdist/runtime files differ, or remote master/tag do not resolve to the attested commit. After upload it waits at most three minutes for the package index and succeeds only if both published artifact hashes match exactly.
