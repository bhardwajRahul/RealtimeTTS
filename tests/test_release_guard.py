"""Regression tests for the mandatory live-artifact release guard."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "release_guard.py"


def _run(*args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *(str(arg) for arg in args)],
        text=True,
        capture_output=True,
        check=False,
    )


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    package = repo / "GuardPkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "init")
    _git(repo, "config", "user.email", "release-guard@example.invalid")
    _git(repo, "config", "user.name", "Release Guard Test")
    _git(repo, "add", "GuardPkg/__init__.py")
    _git(repo, "commit", "-m", "fixture")
    return repo


def _wheel(tmp_path: Path, repo: Path) -> Path:
    version = importlib.metadata.version("pip")
    wheel = tmp_path / f"pip-{version}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.write(repo / "GuardPkg" / "__init__.py", "GuardPkg/__init__.py")
        archive.writestr(
            f"pip-{version}.dist-info/METADATA",
            f"Metadata-Version: 2.1\nName: pip\nVersion: {version}\n",
        )
    return wheel


def test_attest_and_verify_require_exact_runtime_and_artifact(tmp_path):
    repo = _repo(tmp_path)
    wheel = _wheel(tmp_path, repo)
    runtime = tmp_path / "runtime" / "GuardPkg"
    runtime.mkdir(parents=True)
    (runtime / "__init__.py").write_bytes(
        (repo / "GuardPkg" / "__init__.py").read_bytes()
    )
    manifest = tmp_path / "deployment.json"

    attested = _run(
        "attest",
        "--repo",
        repo,
        "--component",
        "fixture",
        "--distribution",
        "pip",
        "--package-dir",
        "GuardPkg",
        "--wheel",
        wheel,
        "--runtime-package-dir",
        runtime,
        "--output",
        manifest,
    )
    assert attested.returncode == 0, attested.stderr
    assert json.loads(manifest.read_text(encoding="utf-8"))["source_commit"]

    verified = _run(
        "verify",
        "--repo",
        repo,
        "--package-dir",
        "GuardPkg",
        "--wheel",
        wheel,
        "--manifest",
        manifest,
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["parity"] == "exact"

    (runtime / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")
    rejected = _run(
        "attest",
        "--repo",
        repo,
        "--component",
        "fixture",
        "--distribution",
        "pip",
        "--package-dir",
        "GuardPkg",
        "--wheel",
        wheel,
        "--runtime-package-dir",
        runtime,
        "--output",
        tmp_path / "rejected.json",
    )
    assert rejected.returncode == 2
    assert "deployed runtime and wheel" in rejected.stderr


def test_dirty_linked_worktree_blocks_release(tmp_path):
    repo = _repo(tmp_path)
    linked = tmp_path / "linked"
    _git(repo, "worktree", "add", "-b", "linked", str(linked))
    (linked / "GuardPkg" / "__init__.py").write_text("VALUE = 9\n", encoding="utf-8")

    result = _run("check-worktrees", "--repo", repo)

    assert result.returncode == 2
    assert "uncommitted files exist in linked worktrees" in result.stderr
    assert str(linked) in result.stderr
