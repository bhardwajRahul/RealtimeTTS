"""End-to-end CLI coverage retained for the release guard."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
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
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "release-guard@example.invalid")
    _git(repo, "config", "user.name", "Release Guard Test")
    _git(repo, "add", "GuardPkg/__init__.py")
    _git(repo, "commit", "--quiet", "-m", "fixture")
    return repo


def _artifacts(tmp_path: Path) -> tuple[Path, Path]:
    wheel = tmp_path / "guard_dist-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("GuardPkg/__init__.py", b"VALUE = 1\r\n")
        archive.writestr(
            "guard_dist-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: guard-dist\nVersion: 1.0\n",
        )

    sdist = tmp_path / "guard_dist-1.0.tar.gz"
    source_root = tmp_path / "sdist" / "guard_dist-1.0" / "GuardPkg"
    source_root.mkdir(parents=True)
    (source_root / "__init__.py").write_bytes(b"VALUE = 1\r\n")
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source_root.parent, arcname="guard_dist-1.0")
    return wheel, sdist


def _runtime(tmp_path: Path) -> tuple[Path, Path]:
    venv = tmp_path / "runtime-venv"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(venv)],
        check=True,
        capture_output=True,
        text=True,
    )
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    purelib = Path(
        subprocess.run(
            [
                str(python),
                "-I",
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    package = purelib / "GuardPkg"
    package.mkdir()
    (package / "__init__.py").write_bytes(b"VALUE = 1\r\n")
    metadata = purelib / "guard_dist-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: guard-dist\nVersion: 1.0\n",
        encoding="utf-8",
    )
    return python, package


def _signing_files(tmp_path: Path) -> tuple[Path, Path]:
    key = tmp_path / "signing_ed25519"
    subprocess.run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "guard-cli-test",
            "-f",
            str(key),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    allowed = tmp_path / "allowed_signers"
    allowed.write_text(
        "guard-cli-test " + key.with_suffix(".pub").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return key, allowed


def test_cli_attests_signed_wheel_sdist_and_current_runtime(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    wheel, sdist = _artifacts(tmp_path)
    runtime_python, runtime_package = _runtime(tmp_path)
    signing_key, allowed_signers = _signing_files(tmp_path)
    manifest = tmp_path / "deployment.json"

    attested = _run(
        "attest",
        "--repo",
        repo,
        "--component",
        "fixture",
        "--distribution",
        "guard-dist",
        "--package-dir",
        "GuardPkg",
        "--wheel",
        wheel,
        "--sdist",
        sdist,
        "--runtime-package-dir",
        runtime_package,
        "--runtime-module",
        "GuardPkg",
        "--runtime-python",
        runtime_python,
        "--signer",
        "guard-cli-test",
        "--signing-key-file",
        signing_key,
        "--output",
        manifest,
    )
    assert attested.returncode == 0, attested.stderr
    assert json.loads(manifest.read_text(encoding="utf-8"))["source_commit"]

    verify_args = (
        "verify",
        "--repo",
        repo,
        "--package-dir",
        "GuardPkg",
        "--wheel",
        wheel,
        "--sdist",
        sdist,
        "--manifest",
        manifest,
        "--signature-file",
        Path(str(manifest) + ".sig"),
        "--allowed-signers-file",
        allowed_signers,
        "--signer",
        "guard-cli-test",
    )
    verified = _run(*verify_args)
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["live_runtime_rechecked"] is True

    (runtime_package / "__init__.py").write_bytes(b"VALUE = 2\r\n")
    rejected = _run(*verify_args)
    assert rejected.returncode == 2
    assert "current runtime differs" in rejected.stderr
