"""Failed Hub requests (downloads and preview ranged reads) become contract issues, offline.

`huggingface_hub.HfFileSystem` and `hf_hub_download` are replaced with fakes; nothing here touches
the network. Exception texts carry a token-like string that must never reach an issue.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from dataset_testkit import hf_source, write_jsonl
from huggingface_hub import errors as hf_errors

from vramforge_estimator.errors import EstimatorError
from vramforge_estimator.inspection import open_rows
from vramforge_estimator.inspection.dataset import inspect_dataset, inspect_dataset_details
from vramforge_estimator.inspection.dataset_files import remote_issue
from vramforge_estimator.schemas import (
    ColumnMapping,
    DatasetFormat,
    DatasetSourceRef,
    ErrorCode,
    Objective,
    Stage,
)
from vramforge_estimator.sources import ResolvedSource, SourceAccess

SECRET = "hf_" + "s" * 34
REF = DatasetSourceRef(reference="acme/demo-set")


def http_error(cls: type[hf_errors.HfHubHTTPError], status: int) -> hf_errors.HfHubHTTPError:
    url = f"https://huggingface.co/datasets/acme/demo-set/resolve/main/x?token={SECRET}"
    response = httpx.Response(status, request=httpx.Request("GET", url))
    return cls(f"{status} for {url}", response=response)


def wrapped(
    cause: BaseException, outer: type[FileNotFoundError] = FileNotFoundError
) -> FileNotFoundError:
    """`raise outer(...) from cause`: how HfFileSystem reports a repository or revision lookup,
    and how hf_hub_download reports a failed metadata request (LocalEntryNotFoundError)."""
    try:
        raise outer(f"datasets/acme/demo-set {SECRET}") from cause
    except FileNotFoundError as exc:
        return exc


LOCAL = hf_errors.LocalEntryNotFoundError


ACCESS = (ErrorCode.SOURCE_ACCESS_DENIED, False, "access_denied")
GONE = (ErrorCode.SOURCE_REVISION_CHANGED, False, "entry_not_found")
RETRY = (ErrorCode.MODEL_METADATA_UNAVAILABLE, True, "network")

MAPPING: list[tuple[BaseException, tuple[ErrorCode, bool, str], int | None]] = [
    (http_error(hf_errors.HfHubHTTPError, 401), ACCESS, 401),
    (http_error(hf_errors.HfHubHTTPError, 403), ACCESS, 403),
    (http_error(hf_errors.GatedRepoError, 403), ACCESS, 403),
    (http_error(hf_errors.RepositoryNotFoundError, 401), ACCESS, 401),
    (wrapped(http_error(hf_errors.RepositoryNotFoundError, 401)), ACCESS, 401),
    (wrapped(http_error(hf_errors.RevisionNotFoundError, 404)), GONE, 404),
    (http_error(hf_errors.RemoteEntryNotFoundError, 404), GONE, 404),
    (FileNotFoundError(f"datasets/acme/demo-set/x.jsonl {SECRET}"), GONE, None),
    (http_error(hf_errors.HfHubHTTPError, 404), RETRY, 404),
    (http_error(hf_errors.HfHubHTTPError, 429), RETRY, 429),
    (http_error(hf_errors.HfHubHTTPError, 500), RETRY, 500),
    (http_error(hf_errors.HfHubHTTPError, 503), RETRY, 503),
    (httpx.ConnectError(f"refused {SECRET}"), RETRY, None),
    (httpx.ReadTimeout(f"timed out {SECRET}"), RETRY, None),
    (TimeoutError(SECRET), RETRY, None),
    (ConnectionResetError(SECRET), RETRY, None),
    (hf_errors.LocalEntryNotFoundError(f"not cached {SECRET}"), RETRY, None),
    # hf_hub_download: "cannot locate the file on the Hub" chained to the failed HEAD request.
    (wrapped(httpx.ConnectError(f"refused {SECRET}"), LOCAL), RETRY, None),
    (wrapped(http_error(hf_errors.HfHubHTTPError, 502), LOCAL), RETRY, 502),
    (wrapped(http_error(hf_errors.HfHubHTTPError, 403), LOCAL), ACCESS, 403),
    (wrapped(http_error(hf_errors.RevisionNotFoundError, 404), LOCAL), GONE, 404),
]


@pytest.mark.parametrize("request_kind", ["download", "read"])
@pytest.mark.parametrize(("exc", "expected", "status"), MAPPING)
def test_remote_issue_mapping(
    exc: BaseException,
    expected: tuple[ErrorCode, bool, str],
    status: int | None,
    request_kind: str,
) -> None:
    issue = remote_issue(exc, "data/train.jsonl", request_kind)  # type: ignore[arg-type]
    assert (issue.code, issue.retryable, issue.details["reason"]) == expected
    assert issue.details["request"] == request_kind
    assert issue.details["file"] == "data/train.jsonl"
    assert issue.details.get("http_status") == status
    assert issue.stage == Stage.INSPECTING and issue.affected_component == "dataset"
    assert SECRET not in issue.model_dump_json()  # exception text is never copied


def test_network_messages_are_dataset_specific_and_name_the_request() -> None:
    download = remote_issue(httpx.ConnectError("x"), "a.jsonl", "download", Stage.TOKENIZING)
    read = remote_issue(httpx.ConnectError("x"), "a.jsonl", "read")
    assert download.stage == Stage.TOKENIZING
    assert "데이터셋 파일을 받지 못했습니다" in download.user_message
    assert "데이터셋 파일을 읽지 못했습니다" in read.user_message
    assert "다시 시도" in read.user_message


@pytest.mark.parametrize(
    "exc",
    [
        hf_errors.OfflineModeIsEnabled(SECRET),
        wrapped(hf_errors.OfflineModeIsEnabled(SECRET), LOCAL),  # hf_hub_download, offline
    ],
)
def test_offline_mode_is_not_retryable(exc: BaseException) -> None:
    issue = remote_issue(exc, "a.jsonl", "read")
    assert (issue.code, issue.retryable) == (ErrorCode.MODEL_METADATA_UNAVAILABLE, False)
    assert issue.details["reason"] == "offline_mode"
    assert SECRET not in issue.model_dump_json()


# ---------------------------------------------------------------- fake Hub transport


class FlakyRemote(io.RawIOBase):
    """Serves `data` like an fsspec file, raising `error` once `fail_at` bytes were served."""

    def __init__(self, data: bytes, fail_at: int | None, error: BaseException | None) -> None:
        super().__init__()
        self.data, self.fail_at, self.error, self.pos = data, fail_at, error, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = 0) -> int:
        target = {0: offset, 1: self.pos + offset, 2: len(self.data) + offset}[whence]
        if target < 0:
            raise ValueError("Seek before start of file")  # fsspec's message
        self.pos = target
        return target

    def tell(self) -> int:
        return self.pos

    def read(self, size: int | None = -1) -> bytes:
        end = len(self.data) if size is None or size < 0 else min(len(self.data), self.pos + size)
        if self.fail_at is not None and end > self.fail_at:
            if self.pos >= self.fail_at:
                assert self.error is not None
                raise self.error
            end = self.fail_at
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk


@pytest.fixture
def hub(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Fake `HfFileSystem` / `hf_hub_download` over a local mirror (`hub["root"]`)."""
    import huggingface_hub

    import vramforge_estimator.inspection.dataset_files as files_module

    state: dict[str, Any] = {
        "root": None,
        "open_error": None,
        "fail_at": None,
        "error": None,
        "download_error": None,
        "log": [],
    }

    class FakeFileSystem:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def open(self, path: str, mode: str = "rb", **kwargs: Any) -> FlakyRemote:
            rel_path = path.split("@", 1)[1].split("/", 1)[1]
            state["log"].append(f"open:{rel_path}")
            if state["open_error"] is not None:
                raise state["open_error"]
            data = (state["root"] / rel_path).read_bytes()
            return FlakyRemote(data, state["fail_at"], state["error"])

    def download(*, filename: str, **kwargs: Any) -> str:
        state["log"].append(f"download:{filename}")
        if state["download_error"] is not None:
            raise state["download_error"]
        return str(state["root"] / filename)

    monkeypatch.setattr(huggingface_hub, "HfFileSystem", FakeFileSystem)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(files_module, "PREVIEW_DOWNLOAD_MAX", 1024)  # preview with ranged reads
    return state


def jsonl_repo(hub: dict[str, Any], root: Path, rows: int = 400) -> ResolvedSource:
    write_jsonl(root / "train.jsonl", [{"text": f"row {i} " + "x" * 50} for i in range(rows)])
    hub["root"] = root
    return hf_source(root)


def test_a_read_failure_after_rows_is_a_retryable_preview_issue(
    tmp_path: Path, hub: dict[str, Any]
) -> None:
    source = jsonl_repo(hub, tmp_path / "repo")
    hub["fail_at"], hub["error"] = 6000, http_error(hf_errors.HfHubHTTPError, 503)
    access = SourceAccess(max_metadata_bytes=16 * 1024)
    details = inspect_dataset_details(source, REF, access, Objective.SFT)
    assert hub["log"] == ["open:train.jsonl"]
    assert details.preview is not None and 0 < len(details.preview.rows) < 100
    result = details.inspection
    assert result.suggested_mapping == ColumnMapping(format=DatasetFormat.TEXT, text="text")
    assert [i.code for i in result.issues] == [ErrorCode.MODEL_METADATA_UNAVAILABLE]
    issue = result.issues[0]
    assert issue.retryable and issue.details["request"] == "read"
    assert (issue.details["http_status"], issue.details["file"]) == (503, "train.jsonl")
    assert ErrorCode.INTERNAL_ERROR not in [i.code for i in result.issues]
    assert SECRET not in result.model_dump_json()


@pytest.mark.parametrize(
    ("error", "code", "retryable"),
    [
        (httpx.ReadTimeout(f"timed out {SECRET}"), ErrorCode.MODEL_METADATA_UNAVAILABLE, True),
        (http_error(hf_errors.HfHubHTTPError, 403), ErrorCode.SOURCE_ACCESS_DENIED, False),
    ],
)
def test_a_read_failure_before_any_column_fails_the_inspection(
    tmp_path: Path, hub: dict[str, Any], error: BaseException, code: ErrorCode, retryable: bool
) -> None:
    # Nothing could be previewed: asking for a column mapping would be pointless.
    source = jsonl_repo(hub, tmp_path / "repo")
    hub["fail_at"], hub["error"] = 0, error
    with pytest.raises(EstimatorError) as excinfo:
        inspect_dataset(source, REF, SourceAccess(), Objective.SFT)
    issue = excinfo.value.issue
    assert (issue.code, issue.retryable, issue.details["request"]) == (code, retryable, "read")
    assert SECRET not in issue.model_dump_json()


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (http_error(hf_errors.HfHubHTTPError, 401), ErrorCode.SOURCE_ACCESS_DENIED),
        (
            wrapped(http_error(hf_errors.RepositoryNotFoundError, 401)),
            ErrorCode.SOURCE_ACCESS_DENIED,
        ),
        (httpx.ConnectError(f"refused {SECRET}"), ErrorCode.MODEL_METADATA_UNAVAILABLE),
    ],
)
def test_open_failures_of_the_remote_file(
    tmp_path: Path, hub: dict[str, Any], error: BaseException, code: ErrorCode
) -> None:
    source = jsonl_repo(hub, tmp_path / "repo")
    hub["open_error"] = error
    with pytest.raises(EstimatorError) as excinfo:
        inspect_dataset(source, REF, SourceAccess(), Objective.SFT)
    assert excinfo.value.issue.code == code
    assert excinfo.value.issue.details["request"] == "read"


def parquet_repo(hub: dict[str, Any], root: Path) -> ResolvedSource:
    root.mkdir(parents=True)
    table = pa.table({"prompt": [f"p{i}" * 20 for i in range(200)], "completion": ["c"] * 200})
    pq.write_table(table, root / "train.parquet", compression="none")
    hub["root"] = root
    return hf_source(root)


@pytest.mark.parametrize(
    "error",
    [
        # An OSError subclass: the Parquet reader would have called the file corrupt.
        http_error(hf_errors.HfHubHTTPError, 503),
        httpx.RemoteProtocolError(f"peer closed {SECRET}"),
    ],
)
def test_a_parquet_read_failure_is_not_reported_as_a_broken_file(
    tmp_path: Path, hub: dict[str, Any], error: BaseException
) -> None:
    source = parquet_repo(hub, tmp_path / "repo")
    hub["fail_at"], hub["error"] = 0, error
    with pytest.raises(EstimatorError) as excinfo:
        inspect_dataset(source, REF, SourceAccess(), Objective.SFT)
    issue = excinfo.value.issue
    assert (issue.code, issue.retryable) == (ErrorCode.MODEL_METADATA_UNAVAILABLE, True)
    assert issue.details["error_type"] == type(error).__name__


def test_a_remote_parquet_preview_without_failures(tmp_path: Path, hub: dict[str, Any]) -> None:
    source = parquet_repo(hub, tmp_path / "repo")
    result = inspect_dataset(source, REF, SourceAccess(), Objective.SFT)
    assert hub["log"] == ["open:train.parquet"]
    assert result.issues == []
    assert result.suggested_mapping == ColumnMapping(
        format=DatasetFormat.PROMPT_COMPLETION, prompt="prompt", completion="completion"
    )


def test_corrupt_remote_bytes_stay_a_data_problem(tmp_path: Path, hub: dict[str, Any]) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "train.parquet").write_bytes(b"PAR1" + b"\x00" * 4000 + b"\xff\xff\xff\x7fPAR1")
    hub["root"] = root
    result = inspect_dataset(hf_source(root), REF, SourceAccess(), Objective.SFT)
    assert [i.code for i in result.issues] == [ErrorCode.DATASET_FORMAT_UNSUPPORTED]
    assert result.issues[0].details["reason"] == "parquet_unreadable"


def test_a_failed_scan_download_stops_the_stream_retryably(
    tmp_path: Path, hub: dict[str, Any]
) -> None:
    source = jsonl_repo(hub, tmp_path / "repo", rows=3)
    hub["download_error"] = httpx.ConnectError(f"refused {SECRET}")
    stream = open_rows(source, config=None, split="train", access=SourceAccess())
    assert list(stream) == []
    assert not stream.complete
    issue = stream.issues[-1]
    assert (issue.code, issue.retryable) == (ErrorCode.MODEL_METADATA_UNAVAILABLE, True)
    assert (issue.stage, issue.details["request"]) == (Stage.TOKENIZING, "download")
    assert "받지 못했습니다" in issue.user_message
    assert SECRET not in issue.model_dump_json()


def test_a_denied_preview_download_fails_the_inspection(
    tmp_path: Path, hub: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import vramforge_estimator.inspection.dataset_files as files_module

    monkeypatch.setattr(files_module, "PREVIEW_DOWNLOAD_MAX", 16 << 20)  # small file: download
    source = jsonl_repo(hub, tmp_path / "repo", rows=3)
    hub["download_error"] = http_error(hf_errors.GatedRepoError, 403)
    with pytest.raises(EstimatorError) as excinfo:
        inspect_dataset(source, REF, SourceAccess(), Objective.SFT)
    issue = excinfo.value.issue
    assert (issue.code, issue.retryable) == (ErrorCode.SOURCE_ACCESS_DENIED, False)
    assert (issue.details["request"], issue.details["http_status"]) == ("download", 403)
    assert hub["log"] == ["download:train.jsonl"]


def test_a_failed_card_download_is_retryable(tmp_path: Path, hub: dict[str, Any]) -> None:
    root = tmp_path / "repo"
    write_jsonl(root / "train.jsonl", [{"text": "t"}])
    (root / "README.md").write_text("---\nlicense: mit\n---\n", encoding="utf-8")
    hub["root"] = root
    hub["download_error"] = http_error(hf_errors.HfHubHTTPError, 502)
    with pytest.raises(EstimatorError) as excinfo:
        inspect_dataset(hf_source(root), REF, SourceAccess(), Objective.SFT)
    issue = excinfo.value.issue
    assert (issue.code, issue.retryable) == (ErrorCode.MODEL_METADATA_UNAVAILABLE, True)
    assert issue.details["file"] == "README.md"
