"""An in-memory `HubClient` for offline tests of the Hugging Face code paths.

Small files are served from bytes written to a temporary directory; safetensors headers are
served from header mappings (the same shape `HfHubClient.read_safetensors_header` returns).
File entries carry git blob ids (sha1 of "blob <size>\\0" + content) like the real Hub, and an
LFS sha256 for files flagged as LFS.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vramforge_estimator.schemas import FileEntry
from vramforge_estimator.sources.hub import HubRepoInfo


def git_blob_id(content: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(content) + content, usedforsecurity=False).hexdigest()


@dataclass
class FakeRepo:
    repo_id: str
    sha: str
    files: dict[str, bytes] = field(default_factory=dict)
    # filename -> (header mapping, file size, lfs sha256)
    shards: dict[str, tuple[dict[str, Any], int, str]] = field(default_factory=dict)
    lfs: frozenset[str] = frozenset()
    private: bool = False
    gated: bool = False
    last_modified: str | None = "2026-09-22T03:52:45+00:00"
    aliases: tuple[str, ...] = ()
    revisions: tuple[str, ...] = ("main",)

    def entries(self) -> tuple[FileEntry, ...]:
        out = [
            FileEntry(
                path=name,
                size=len(content),
                sha256=hashlib.sha256(content).hexdigest() if name in self.lfs else None,
                blob_id=git_blob_id(content),
            )
            for name, content in self.files.items()
        ]
        out += [
            FileEntry(path=name, size=size, sha256=lfs_sha, blob_id="0" * 40)
            for name, (_, size, lfs_sha) in self.shards.items()
        ]
        return tuple(sorted(out, key=lambda e: e.path))


class FakeHub:
    def __init__(self, root: Path, repos: Mapping[tuple[str, str], FakeRepo]) -> None:
        self.root = root
        self.repos = dict(repos)
        self.calls: list[tuple[str, str]] = []
        self.errors: dict[str, BaseException] = {}  # method name -> exception to raise
        self.tamper: dict[str, bytes] = {}  # filename -> bytes served instead of the real ones

    def _repo(self, kind: str, repo_id: str) -> FakeRepo:
        for (repo_kind, _), repo in self.repos.items():
            if repo_kind == kind and repo_id in (repo.repo_id, *repo.aliases):
                return repo
        raise KeyError(repo_id)

    def _maybe_fail(self, method: str) -> None:
        if method in self.errors:
            raise self.errors[method]

    def repo_info(self, kind: str, repo_id: str, revision: str | None) -> HubRepoInfo:
        self.calls.append(("repo_info", repo_id))
        self._maybe_fail("repo_info")
        repo = self._repo(kind, repo_id)
        if revision not in (None, repo.sha, *repo.revisions):
            raise KeyError(revision)
        return HubRepoInfo(
            repo_id=repo.repo_id,
            sha=repo.sha,
            files=repo.entries(),
            private=repo.private,
            gated=repo.gated,
            last_modified=repo.last_modified,
        )

    def check_file_access(self, kind: str, repo_id: str, revision: str, filename: str) -> None:
        self.calls.append(("check_file_access", filename))
        self._maybe_fail("check_file_access")

    def download_file(self, kind: str, repo_id: str, revision: str, filename: str) -> Path:
        self.calls.append(("download_file", filename))
        self._maybe_fail("download_file")
        repo = self._repo(kind, repo_id)
        assert revision == repo.sha, "downloads must use the pinned commit"
        content = self.tamper.get(filename, repo.files[filename])
        path = self.root / repo.sha / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def read_safetensors_header(self, repo_id: str, revision: str, filename: str) -> dict[str, Any]:
        self.calls.append(("read_safetensors_header", filename))
        self._maybe_fail("read_safetensors_header")
        repo = self._repo("model", repo_id)
        assert revision == repo.sha, "headers must be read at the pinned commit"
        return repo.shards[filename][0]
