"""Build a deterministic AIC 2026 retrieval stress corpus from remote ZIPs.

The script uses HTTP Range requests, so it downloads only the ZIP directory and
the selected members instead of downloading every multi-gigabyte archive.
It creates 1,200 balanced keyframe samples by default, enriches them with the
official object detections and local video metadata, and writes pseudo-ground-
truth Textual KIS, Q&A, and TRAKE queries.

The generated labels are suitable for stress/regression testing, not for
reporting official model accuracy: object labels and counts come from the
provided Faster R-CNN output rather than manual annotation.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import http.client
import io
import json
import lzma
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.parse import urlsplit


BASE_URL = "https://aic-data.ledo.io.vn"
OBJECTS_URL = f"{BASE_URL}/objects-aic25-b1.zip"
DEFAULT_TOTAL = 1_200
FRAMES_PER_VIDEO = 4
DEFAULT_OBJECT_THRESHOLD = 0.30
HTTP_USER_AGENT = "AICHCM-2026-Glitch stress-corpus-builder/1.0"


@dataclass(frozen=True)
class ArchiveSpec:
    dataset: str
    part: str
    filename: str

    @property
    def url(self) -> str:
        return f"{BASE_URL}/{self.filename}"

    @property
    def key(self) -> str:
        return f"{self.dataset}{self.part}"


ARCHIVES = (
    ArchiveSpec("L21", "", "Keyframes_L21.zip"),
    ArchiveSpec("L22", "", "Keyframes_L22.zip"),
    ArchiveSpec("L23", "", "Keyframes_L23.zip"),
    ArchiveSpec("L24", "", "Keyframes_L24.zip"),
    ArchiveSpec("L25", "", "Keyframes_L25.zip"),
    ArchiveSpec("L26", "_a", "Keyframes_L26_a.zip"),
    ArchiveSpec("L26", "_b", "Keyframes_L26_b.zip"),
    ArchiveSpec("L26", "_c", "Keyframes_L26_c.zip"),
    ArchiveSpec("L26", "_d", "Keyframes_L26_d.zip"),
    ArchiveSpec("L26", "_e", "Keyframes_L26_e.zip"),
    ArchiveSpec("L27", "", "Keyframes_L27.zip"),
    ArchiveSpec("L28", "", "Keyframes_L28.zip"),
    ArchiveSpec("L29", "", "Keyframes_L29.zip"),
    ArchiveSpec("L30", "", "Keyframes_L30.zip"),
)

IMAGE_MEMBER_RE = re.compile(
    r"(?:^|/)(?P<video>L\d+_V\d+)/(?P<stem>\d+)\.(?P<ext>jpe?g|png)$",
    re.IGNORECASE,
)
OBJECT_MEMBER_RE = re.compile(
    r"(?:^|/)(?P<video>L\d+_V\d+)/(?P<stem>\d+)\.json$",
    re.IGNORECASE,
)
VIDEO_ID_RE = re.compile(r"^L\d+_V\d+$", re.IGNORECASE)


VI_LABELS = {
    "person": "người",
    "man": "người đàn ông",
    "woman": "người phụ nữ",
    "boy": "bé trai",
    "girl": "bé gái",
    "child": "trẻ em",
    "car": "ô tô",
    "bus": "xe buýt",
    "truck": "xe tải",
    "motorcycle": "xe máy",
    "bicycle": "xe đạp",
    "traffic light": "đèn giao thông",
    "traffic sign": "biển báo giao thông",
    "television": "ti vi",
    "mobile phone": "điện thoại",
    "laptop": "máy tính xách tay",
    "computer keyboard": "bàn phím",
    "table": "bàn",
    "chair": "ghế",
    "food": "đồ ăn",
    "bottle": "chai",
    "cup": "cốc",
    "book": "sách",
    "dog": "chó",
    "cat": "mèo",
    "tree": "cây",
    "building": "tòa nhà",
    "flower": "hoa",
    "human face": "khuôn mặt người",
    "clothing": "quần áo",
    "footwear": "giày dép",
    "plant": "cây cối",
    "window": "cửa sổ",
    "boat": "thuyền",
    "flag": "lá cờ",
    "fruit": "trái cây",
    "tomato": "cà chua",
    "wheel": "bánh xe",
    "land vehicle": "phương tiện đường bộ",
    "dinosaur": "khủng long",
    "fish": "cá",
}


def log(message: str) -> None:
    print(message, flush=True)


class HTTPRangeReader(io.RawIOBase):
    """Seekable, read-only HTTP file backed by byte-range requests."""

    def __init__(
        self,
        url: str,
        *,
        timeout: float = 90.0,
        block_size: int = 256 * 1024,
        max_cached_blocks: int = 48,
        retries: int = 4,
    ) -> None:
        super().__init__()
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(f"Only HTTPS URLs are supported: {url}")
        self.url = url
        self.host = parsed.hostname
        self.port = parsed.port or 443
        self.path = parsed.path or "/"
        if parsed.query:
            self.path += f"?{parsed.query}"
        self.timeout = timeout
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks
        self.retries = retries
        self._position = 0
        self._connection: http.client.HTTPSConnection | None = None
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        self._length = self._discover_length()

    def _new_connection(self) -> http.client.HTTPSConnection:
        return http.client.HTTPSConnection(self.host, self.port, timeout=self.timeout)

    def _request(self, method: str, headers: dict[str, str]) -> tuple[int, Any, bytes]:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                if self._connection is None:
                    self._connection = self._new_connection()
                request_headers = {
                    "User-Agent": HTTP_USER_AGENT,
                    "Accept-Encoding": "identity",
                    **headers,
                }
                self._connection.request(method, self.path, headers=request_headers)
                response = self._connection.getresponse()
                body = response.read()
                return response.status, response.headers, body
            except Exception as exc:  # network reset/timeout retry boundary
                last_error = exc
                self._drop_connection()
                if attempt < self.retries:
                    time.sleep(min(2 ** (attempt - 1), 5))
        raise OSError(f"HTTP request failed for {self.url}: {last_error}") from last_error

    def _drop_connection(self) -> None:
        if self._connection is not None:
            try:
                self._connection.close()
            finally:
                self._connection = None

    def _discover_length(self) -> int:
        result = subprocess.run(
            [
                "curl.exe",
                "-sSIL",
                "--max-time",
                str(max(30, int(self.timeout))),
                self.url,
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise OSError(
                f"curl HEAD failed for {self.url}: "
                f"{result.stderr.decode('utf-8', errors='replace').strip()}"
            )
        header_text = result.stdout.decode("iso-8859-1", errors="replace")
        lengths = re.findall(r"(?im)^Content-Length:\s*(\d+)\s*$", header_text)
        if not lengths:
            raise OSError(f"Missing Content-Length for {self.url}")
        if not re.search(r"(?im)^Accept-Ranges:\s*bytes\s*$", header_text):
            raise OSError(f"Server does not advertise byte ranges for {self.url}")
        return int(lengths[-1])

    @property
    def length(self) -> int:
        return self._length

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            position = offset
        elif whence == os.SEEK_CUR:
            position = self._position + offset
        elif whence == os.SEEK_END:
            position = self._length + offset
        else:
            raise ValueError(f"Unsupported whence: {whence}")
        if position < 0:
            raise ValueError("Negative seek position")
        self._position = min(position, self._length)
        return self._position

    def _fetch_range(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self._length:
            raise ValueError(f"Invalid byte range {start}-{end} for {self._length}")
        expected = end - start + 1
        last_error = "unknown error"
        for attempt in range(1, self.retries + 1):
            # Retry in Python rather than curl. curl can concatenate a partial
            # failed body with its retry body when stdout is captured, which
            # corrupts ZIP ranges even though the final HTTP response succeeds.
            result = subprocess.run(
                [
                    "curl.exe",
                    "-sS",
                    "--fail",
                    "--location",
                    "--max-time",
                    str(max(30, int(self.timeout))),
                    "--range",
                    f"{start}-{end}",
                    self.url,
                ],
                check=False,
                capture_output=True,
            )
            if result.returncode == 0 and len(result.stdout) == expected:
                return result.stdout
            if result.returncode != 0:
                last_error = result.stderr.decode("utf-8", errors="replace").strip()
            else:
                last_error = f"expected {expected} bytes, got {len(result.stdout)}"
            if attempt < self.retries:
                time.sleep(min(2 ** (attempt - 1), 5))
        raise OSError(
            f"curl range {start}-{end} failed for {self.url}: {last_error}"
        )

    def _get_block(self, block_index: int) -> bytes:
        cached = self._cache.get(block_index)
        if cached is not None:
            self._cache.move_to_end(block_index)
            return cached
        start = block_index * self.block_size
        end = min(self._length - 1, start + self.block_size - 1)
        data = self._fetch_range(start, end)
        self._cache[block_index] = data
        self._cache.move_to_end(block_index)
        while len(self._cache) > self.max_cached_blocks:
            self._cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._length:
            return b""
        if size is None or size < 0:
            size = self._length - self._position
        size = min(size, self._length - self._position)
        if size == 0:
            return b""

        # ZIP central directories can be tens of MB. Fetch large sequential
        # reads directly rather than issuing hundreds of block requests.
        if size >= self.block_size * 4:
            start = self._position
            end = start + size - 1
            data = self._fetch_range(start, end)
            self._position += len(data)
            return data

        output = bytearray()
        remaining = size
        while remaining:
            block_index = self._position // self.block_size
            block = self._get_block(block_index)
            block_offset = self._position % self.block_size
            available = min(remaining, len(block) - block_offset)
            if available <= 0:
                break
            output.extend(block[block_offset : block_offset + available])
            self._position += available
            remaining -= available
        return bytes(output)

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)

    def close(self) -> None:
        self._drop_connection()
        self._cache.clear()
        super().close()


class RemoteZip:
    def __init__(self, url: str, *, timeout: float = 90.0) -> None:
        self.reader = HTTPRangeReader(url, timeout=timeout)
        self.archive = zipfile.ZipFile(self.reader)

    def __enter__(self) -> zipfile.ZipFile:
        return self.archive

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.archive.close()
        self.reader.close()


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    values = [round(i * (length - 1) / (count - 1)) for i in range(count)]
    # round() can theoretically collide; fill deterministically if it does.
    unique = list(dict.fromkeys(values))
    if len(unique) < count:
        for index in range(length):
            if index not in unique:
                unique.append(index)
                if len(unique) == count:
                    break
        unique.sort()
    return unique


def interior_spaced_indices(
    length: int, count: int, *, edge_fraction: float = 0.08
) -> list[int]:
    """Sample temporal positions while avoiding common intro/outro frames."""
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    start = min(length - 1, max(0, round((length - 1) * edge_fraction)))
    end = max(start, min(length - 1, round((length - 1) * (1 - edge_fraction))))
    span = end - start + 1
    if span < count:
        return evenly_spaced_indices(length, count)
    return [start + index for index in evenly_spaced_indices(span, count)]


def allocate_quotas(total: int) -> dict[str, int]:
    if total < len(ARCHIVES):
        raise ValueError(f"--total must be at least {len(ARCHIVES)}")
    datasets = [f"L{number}" for number in range(21, 31)]
    base, remainder = divmod(total, len(datasets))
    dataset_quotas = {
        dataset: base + (1 if index < remainder else 0)
        for index, dataset in enumerate(datasets)
    }
    quotas: dict[str, int] = {}
    for dataset in datasets:
        matching = [spec for spec in ARCHIVES if spec.dataset == dataset]
        part_base, part_remainder = divmod(dataset_quotas[dataset], len(matching))
        for index, spec in enumerate(matching):
            quotas[spec.key] = part_base + (1 if index < part_remainder else 0)
    return quotas


def image_members(archive: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    grouped: dict[str, list[zipfile.ZipInfo]] = defaultdict(list)
    for info in archive.infolist():
        match = IMAGE_MEMBER_RE.search(info.filename.replace("\\", "/"))
        if match:
            grouped[match.group("video").upper()].append(info)
    for infos in grouped.values():
        infos.sort(key=member_frame_number)
    return dict(sorted(grouped.items()))


def member_frame_number(info: zipfile.ZipInfo) -> int:
    match = IMAGE_MEMBER_RE.search(info.filename.replace("\\", "/"))
    if not match:
        raise ValueError(f"Not an image member: {info.filename}")
    return int(match.group("stem"))


def select_members(
    grouped: dict[str, list[zipfile.ZipInfo]],
    quota: int,
    *,
    frames_per_video: int = FRAMES_PER_VIDEO,
) -> list[zipfile.ZipInfo]:
    if quota <= 0:
        return []
    videos = [video for video, infos in grouped.items() if infos]
    if not videos:
        raise ValueError("Archive contains no recognised keyframe members")

    target_videos = min(len(videos), math.ceil(quota / frames_per_video))
    chosen_videos = [videos[index] for index in evenly_spaced_indices(len(videos), target_videos)]
    base, remainder = divmod(quota, target_videos)
    selected: list[zipfile.ZipInfo] = []
    selected_names: set[str] = set()
    for index, video in enumerate(chosen_videos):
        requested = base + (1 if index < remainder else 0)
        infos = grouped[video]
        for position in interior_spaced_indices(
            len(infos), min(requested, len(infos))
        ):
            info = infos[position]
            selected.append(info)
            selected_names.add(info.filename)

    if len(selected) < quota:
        all_infos = [info for video in videos for info in grouped[video]]
        remaining = [info for info in all_infos if info.filename not in selected_names]
        need = min(quota - len(selected), len(remaining))
        for index in evenly_spaced_indices(len(remaining), need):
            selected.append(remaining[index])

    if len(selected) != quota:
        raise ValueError(
            f"Could only select {len(selected)} of {quota} requested keyframes"
        )
    return sorted(
        selected,
        key=lambda info: (
            IMAGE_MEMBER_RE.search(info.filename.replace("\\", "/")).group("video"),
            member_frame_number(info),
        ),
    )


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(data)
    os.replace(temporary, path)


@dataclass(frozen=True)
class RangeTransfer:
    start: int
    end: int
    output: Path

    @property
    def expected_size(self) -> int:
        return self.end - self.start + 1


def run_parallel_range_transfers(
    url: str,
    transfers: list[RangeTransfer],
    *,
    timeout: float,
    parallel: int,
    batch_size: int,
    retries: int = 4,
) -> None:
    """Download independent byte ranges with curl's parallel transfer engine."""
    pending = list(transfers)
    for transfer in pending:
        transfer.output.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        failed: list[RangeTransfer] = []
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            for transfer in batch:
                if transfer.output.exists():
                    transfer.output.unlink()
            command = [
                "curl.exe",
                "-sS",
                "--parallel",
                "--parallel-immediate",
                "--parallel-max",
                str(parallel),
            ]
            for index, transfer in enumerate(batch):
                if index:
                    command.append("--next")
                command.extend(
                    [
                        "--fail",
                        "--location",
                        "--max-time",
                        str(max(30, int(timeout))),
                        "--range",
                        f"{transfer.start}-{transfer.end}",
                        "--output",
                        str(transfer.output),
                        url,
                    ]
                )
            result = subprocess.run(command, check=False, capture_output=True)
            for transfer in batch:
                if (
                    not transfer.output.exists()
                    or transfer.output.stat().st_size != transfer.expected_size
                ):
                    failed.append(transfer)
            if result.returncode != 0 and not failed:
                raise OSError(
                    f"curl parallel transfer failed for {url}: "
                    f"{result.stderr.decode('utf-8', errors='replace').strip()}"
                )
        if not failed:
            return
        pending = failed
        if attempt < retries:
            log(
                f"  retrying {len(pending)} incomplete ranges "
                f"(attempt {attempt + 1}/{retries})"
            )
            time.sleep(min(2 ** (attempt - 1), 5))
    details = ", ".join(
        f"{item.start}-{item.end}" for item in pending[:5]
    )
    raise OSError(f"Failed to download {len(pending)} ranges from {url}: {details}")


def parse_local_header(header: bytes, info: zipfile.ZipInfo) -> int:
    if len(header) < 30:
        raise zipfile.BadZipFile(f"Short local header for {info.filename}")
    (
        signature,
        _version,
        _flags,
        compression,
        _mtime,
        _mdate,
        _crc,
        _compressed_size,
        _file_size,
        filename_length,
        extra_length,
    ) = struct.unpack("<IHHHHHIIIHH", header[:30])
    if signature != 0x04034B50:
        raise zipfile.BadZipFile(f"Invalid local header for {info.filename}")
    if compression != info.compress_type:
        raise zipfile.BadZipFile(f"Compression mismatch for {info.filename}")
    return info.header_offset + 30 + filename_length + extra_length


def decompress_member(info: zipfile.ZipInfo, compressed: bytes) -> bytes:
    if info.compress_type == zipfile.ZIP_STORED:
        data = compressed
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(compressed, -15)
    elif info.compress_type == zipfile.ZIP_BZIP2:
        data = bz2.decompress(compressed)
    elif info.compress_type == zipfile.ZIP_LZMA:
        data = lzma.decompress(compressed)
    else:
        raise NotImplementedError(
            f"Unsupported ZIP compression {info.compress_type} for {info.filename}"
        )
    if len(data) != info.file_size:
        raise zipfile.BadZipFile(
            f"Size mismatch for {info.filename}: expected {info.file_size}, got {len(data)}"
        )
    if (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
        raise zipfile.BadZipFile(f"CRC mismatch for {info.filename}")
    return data


def batch_extract_remote_members(
    url: str,
    members: list[tuple[zipfile.ZipInfo, Path]],
    *,
    scratch_root: Path,
    timeout: float,
    parallel: int,
    batch_size: int,
) -> None:
    """Extract selected ZIP members with two parallel HTTP range passes."""
    pending = [
        (info, destination)
        for info, destination in members
        if not destination.exists() or destination.stat().st_size == 0
    ]
    if not pending:
        return
    archive_key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    scratch = scratch_root / archive_key
    scratch.mkdir(parents=True, exist_ok=True)

    header_paths: dict[str, Path] = {}
    header_transfers = []
    for index, (info, _) in enumerate(pending):
        path = scratch / f"{index:05d}.header"
        header_paths[info.filename] = path
        header_transfers.append(
            RangeTransfer(info.header_offset, info.header_offset + 63, path)
        )
    run_parallel_range_transfers(
        url,
        header_transfers,
        timeout=timeout,
        parallel=parallel,
        batch_size=batch_size,
    )

    compressed_paths: dict[str, Path] = {}
    data_transfers = []
    zero_length: set[str] = set()
    for index, (info, _) in enumerate(pending):
        data_offset = parse_local_header(header_paths[info.filename].read_bytes(), info)
        path = scratch / f"{index:05d}.compressed"
        compressed_paths[info.filename] = path
        if info.compress_size == 0:
            zero_length.add(info.filename)
            path.write_bytes(b"")
        else:
            data_transfers.append(
                RangeTransfer(
                    data_offset,
                    data_offset + info.compress_size - 1,
                    path,
                )
            )
    run_parallel_range_transfers(
        url,
        data_transfers,
        timeout=timeout,
        parallel=parallel,
        batch_size=batch_size,
    )

    for info, destination in pending:
        compressed = compressed_paths[info.filename].read_bytes()
        data = decompress_member(info, compressed)
        atomic_write_bytes(destination, data)
    shutil.rmtree(scratch)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_image_bytes(data: bytes, suffix: str) -> None:
    suffix = suffix.casefold()
    if suffix in {".jpg", ".jpeg"} and not data.startswith(b"\xff\xd8\xff"):
        raise ValueError("Downloaded member is not a JPEG")
    if suffix == ".png" and not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Downloaded member is not a PNG")


def parse_image_identity(info: zipfile.ZipInfo) -> tuple[str, int, str]:
    match = IMAGE_MEMBER_RE.search(info.filename.replace("\\", "/"))
    if not match:
        raise ValueError(info.filename)
    return match.group("video").upper(), int(match.group("stem")), match.group("ext").lower()


def load_frame_map(repo_root: Path, video_id: str) -> dict[int, dict[str, Any]]:
    path = repo_root / "data" / "map-keyframes" / f"{video_id}.csv"
    if not path.exists():
        return {}
    mapping: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                n = int(row["n"])
                mapping[n] = {
                    "pts_time": float(row["pts_time"]),
                    "fps": float(row["fps"]),
                    "frame_idx": int(row["frame_idx"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
    return mapping


def load_media_info(repo_root: Path, video_id: str) -> dict[str, Any]:
    path = repo_root / "data" / "media-info" / f"{video_id}.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def build_object_member_index(
    archive: zipfile.ZipFile,
) -> dict[tuple[str, int], zipfile.ZipInfo]:
    index: dict[tuple[str, int], zipfile.ZipInfo] = {}
    for info in archive.infolist():
        match = OBJECT_MEMBER_RE.search(info.filename.replace("\\", "/"))
        if match:
            index[(match.group("video").upper(), int(match.group("stem")))] = info
    return index


def normalise_label(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().casefold().replace("_", " ").split())


def first_list(mapping: dict[str, Any], keys: Iterable[str]) -> list[Any]:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, list):
            return value
    return []


def parse_detection_dict(item: dict[str, Any]) -> dict[str, Any] | None:
    label = normalise_label(
        item.get("label")
        or item.get("class_name")
        or item.get("name")
        or item.get("entity")
    )
    if not label:
        return None
    raw_score = item.get("score", item.get("confidence", 1.0))
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    box = item.get("bbox") or item.get("box") or item.get("detection_box")
    return {"label": label, "score": score, "bbox_raw": box}


def parse_detections(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        output = []
        for item in raw:
            if isinstance(item, dict):
                detection = parse_detection_dict(item)
                if detection:
                    output.append(detection)
        return output
    if not isinstance(raw, dict):
        return []

    for key in ("objects", "detections", "predictions", "results"):
        value = raw.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            return [
                detection
                for item in value
                if isinstance(item, dict)
                for detection in [parse_detection_dict(item)]
                if detection is not None
            ]

    labels = first_list(
        raw,
        (
            "detection_class_entities",
            "detection_class_names",
            "class_entities",
            "labels",
            "classes",
        ),
    )
    scores = first_list(raw, ("detection_scores", "scores", "confidences"))
    boxes = first_list(raw, ("detection_boxes", "boxes", "bboxes"))
    output = []
    for index, label_value in enumerate(labels):
        label = normalise_label(label_value)
        if not label:
            continue
        try:
            score = float(scores[index]) if index < len(scores) else 1.0
        except (TypeError, ValueError):
            score = 0.0
        box = boxes[index] if index < len(boxes) else None
        output.append({"label": label, "score": score, "bbox_raw": box})
    return output


def canonical_box(raw_box: Any) -> list[float] | None:
    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None
    try:
        values = [float(value) for value in raw_box]
    except (TypeError, ValueError):
        return None
    if not all(0.0 <= value <= 1.05 for value in values):
        return None
    # TensorFlow Object Detection exports [ymin, xmin, ymax, xmax].
    ymin, xmin, ymax, xmax = values
    if xmin >= xmax or ymin >= ymax:
        return None
    return [xmin, ymin, xmax, ymax]


def summarise_detections(
    raw: Any, threshold: float
) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    detections = []
    for detection in parse_detections(raw):
        if detection["score"] < threshold:
            continue
        detections.append(
            {
                "label": detection["label"],
                "score": round(float(detection["score"]), 6),
                "bbox": canonical_box(detection.get("bbox_raw")),
            }
        )
    detections.sort(key=lambda item: (-item["score"], item["label"]))
    counts = dict(sorted(Counter(item["label"] for item in detections).items()))
    best_score: dict[str, float] = {}
    for item in detections:
        best_score[item["label"]] = max(best_score.get(item["label"], 0.0), item["score"])
    labels = sorted(best_score, key=lambda label: (-best_score[label], label))
    return detections, counts, labels


def dedupe_phrases(values: Iterable[Any], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        phrase = " ".join(str(value).strip().split()) if value is not None else ""
        key = phrase.casefold()
        if phrase and key not in seen:
            seen.add(key)
            output.append(phrase)
            if len(output) == limit:
                break
    return output


def video_keywords(media: dict[str, Any]) -> list[str]:
    values: list[Any] = []
    title = media.get("title")
    author = media.get("author")
    if title:
        values.append(title)
    if author:
        values.append(author)
    keywords = media.get("keywords")
    if isinstance(keywords, list):
        values.extend(keywords)
    return dedupe_phrases(values, 12)


def translate_label(label: str) -> str:
    return VI_LABELS.get(label, label)


def human_join(values: list[str]) -> str:
    if not values:
        return "the available video context"
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def valid_range(frame_index: int, radius: int = 5) -> list[int]:
    return [max(0, frame_index - radius), frame_index + radius]


def make_target(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": sample["video_id"],
        "keyframe_index": sample["keyframe_index"],
        "frame_index": sample["source_frame_index"],
        "valid_frame_range": valid_range(sample["source_frame_index"]),
    }


def textual_query_text(sample: dict[str, Any], index: int) -> tuple[str, str]:
    labels = sample.get("object_keywords", [])[:4]
    metadata = sample.get("video_keywords", [])
    context = metadata[0] if metadata else sample["video_id"]
    if not labels:
        clues = metadata[1:4] or [context]
        if index % 2:
            return (
                f"Tìm một khung hình thuộc video có thông tin: {human_join(clues)}. "
                f"Ưu tiên đúng video và đúng mốc thời gian.",
                "vi-metadata-only",
            )
        return (
            f"Find a frame from the video associated with {human_join(clues)}. "
            f"Prioritize the correct video and temporal location.",
            "en-metadata-only",
        )
    if index % 3 == 0:
        return (
            f"Find the moment showing {human_join(labels)}. Video context: {context}.",
            "en",
        )
    if index % 3 == 1:
        translated = [translate_label(label) for label in labels]
        return (
            f"Tìm khoảnh khắc có {human_join(translated)}. Bối cảnh video: {context}.",
            "vi",
        )
    noisy = ", ".join(labels) if labels else "visual scene"
    return (
        f"Need exact frame please - {noisy}; maybe from '{context}'. Ignore unrelated clips.",
        "mixed-noisy",
    )


def generate_queries(samples: list[dict[str, Any]]) -> tuple[list[dict], list[dict], list[dict]]:
    textual: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        query, language = textual_query_text(sample, index)
        textual.append(
            {
                "query_id": f"tkis-{index:04d}",
                "type": "textual_kis",
                "query": query,
                "language": language,
                "keywords": sample["keywords"],
                "difficulty": (
                    "metadata_only"
                    if not sample.get("object_keywords")
                    else ("mixed_noisy" if language == "mixed-noisy" else "object_metadata")
                ),
                "target": make_target(sample),
                "source_sample_id": sample["sample_id"],
                "ground_truth_quality": "pseudo_from_provided_objects_and_metadata",
            }
        )

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_video[sample["video_id"]].append(sample)
    for group in by_video.values():
        group.sort(key=lambda item: item["source_frame_index"])

    qa: list[dict[str, Any]] = []
    for video_id, group in sorted(by_video.items()):
        candidates = [
            sample
            for sample in group
            if sample.get("object_counts")
        ]
        if not candidates:
            continue
        sample = max(
            candidates,
            key=lambda item: (
                max(item["object_counts"].values()),
                len(item["object_counts"]),
            ),
        )
        label, count = max(
            sample["object_counts"].items(), key=lambda pair: (pair[1], pair[0])
        )
        context = (sample.get("video_keywords") or [video_id])[0]
        qa.append(
            {
                "query_id": f"qa-{len(qa) + 1:04d}",
                "type": "qa",
                "query": (
                    f"In the target scene from '{context}', how many detected "
                    f"instances of '{label}' are visible?"
                ),
                "answer": str(count),
                "answer_label": label,
                "keywords": dedupe_phrases([label, *sample["keywords"]], 12),
                "target": make_target(sample),
                "source_sample_id": sample["sample_id"],
                "ground_truth_quality": "pseudo_count_from_provided_object_detector",
            }
        )

    trake: list[dict[str, Any]] = []
    for video_id, group in sorted(by_video.items()):
        if len(group) < 3:
            continue
        selected = [group[index] for index in evenly_spaced_indices(len(group), min(4, len(group)))]
        events = []
        for event_index, sample in enumerate(selected, start=1):
            labels = sample.get("object_keywords", [])[:3]
            if labels:
                event_text = f"moment showing {human_join(labels)}"
            else:
                positions = {
                    1: "opening",
                    2: "early-middle",
                    3: "late-middle",
                    len(selected): "closing",
                }
                metadata = sample.get("video_keywords", [])
                metadata_clues = metadata[1:3] or metadata[:1] or [video_id]
                event_text = (
                    f"{positions.get(event_index, 'intermediate')} sampled moment "
                    f"associated with {human_join(metadata_clues)}"
                )
            events.append(
                {
                    "event_index": event_index,
                    "description": event_text,
                    "frame_index": sample["source_frame_index"],
                    "valid_frame_range": valid_range(sample["source_frame_index"]),
                    "source_sample_id": sample["sample_id"],
                }
            )
        context = (selected[0].get("video_keywords") or [video_id])[0]
        query = (
            f"In the video '{context}', locate these sampled moments in temporal order: "
            + "; ".join(
                f"({event['event_index']}) {event['description']}" for event in events
            )
            + "."
        )
        trake.append(
            {
                "query_id": f"trake-{len(trake) + 1:04d}",
                "type": "trake",
                "query": query,
                "video_id": video_id,
                "events": events,
                "target": {
                    "video_id": video_id,
                    "frame_indices": [event["frame_index"] for event in events],
                    "valid_frame_ranges": [event["valid_frame_range"] for event in events],
                },
                "ground_truth_quality": "pseudo_sequence_from_temporally_spaced_keyframes",
            }
        )
    return textual, qa, trake


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
            count += 1
    return count


def write_manifest_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    columns = [
        "sample_id",
        "dataset",
        "video_id",
        "keyframe_index",
        "source_frame_index",
        "timestamp_seconds",
        "retrieval_frame_id",
        "relative_image_path",
        "keywords",
        "title",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "sample_id": sample["sample_id"],
                    "dataset": sample["dataset"],
                    "video_id": sample["video_id"],
                    "keyframe_index": sample["keyframe_index"],
                    "source_frame_index": sample["source_frame_index"],
                    "timestamp_seconds": sample.get("timestamp_seconds"),
                    "retrieval_frame_id": sample["retrieval_frame_id"],
                    "relative_image_path": sample["relative_image_path"],
                    "keywords": " | ".join(sample["keywords"]),
                    "title": sample.get("title", ""),
                }
            )


def write_markdown(
    path: Path,
    *,
    samples: list[dict[str, Any]],
    textual: list[dict[str, Any]],
    qa: list[dict[str, Any]],
    trake: list[dict[str, Any]],
) -> None:
    datasets = Counter(sample["dataset"] for sample in samples)
    lines = [
        "# AIC 2026 stress-test corpus",
        "",
        "This corpus is a deterministic engineering stress test built from the official "
        "batch-1 keyframes, object detections, frame maps, and video metadata.",
        "",
        "> Important: labels are pseudo-ground truth derived from the provided Faster "
        "R-CNN detections and metadata. They are not human annotations and must not be "
        "reported as official retrieval accuracy.",
        "",
        "## Corpus",
        "",
        f"- Frames: **{len(samples)}**",
        f"- Videos represented: **{len({sample['video_id'] for sample in samples})}**",
        f"- Textual KIS queries: **{len(textual)}**",
        f"- Q&A queries: **{len(qa)}**",
        f"- TRAKE sequences: **{len(trake)}**",
        "- Dataset distribution: " + ", ".join(f"{key}={value}" for key, value in sorted(datasets.items())),
        "",
        "The official preliminary-round scoring accepts at most 100 answers and uses "
        "R@k at k = 1, 5, 20, 50, and 100. The included evaluator follows those cutoffs.",
        "",
        "## Files",
        "",
        "- `manifest.jsonl`: complete frame metadata, keywords, detections, and provenance.",
        "- `manifest.csv`: spreadsheet-friendly frame manifest.",
        "- `queries/textual_kis.jsonl`: one retrieval query per sampled frame.",
        "- `queries/qa.jsonl`: one detector-count question per represented video when possible.",
        "- `queries/trake.jsonl`: ordered 3-4 moment sequences per represented video.",
        "- `queries/all_queries.jsonl`: concatenation of all query types.",
        "- `frames/`: sampled JPEG/PNG files grouped by video ID.",
        "- `objects/`: raw object JSON for each sampled frame when available.",
        "",
        "## Representative queries",
        "",
        "| ID | Type | Query | Target |",
        "|---|---|---|---|",
    ]
    representatives = textual[:20] + qa[:20] + trake[:20]
    for query in representatives:
        target = query.get("target", {})
        if query["type"] == "trake":
            target_text = f"{target.get('video_id')} / {target.get('frame_indices')}"
        else:
            target_text = f"{target.get('video_id')} / {target.get('frame_index')}"
        query_text = query["query"].replace("|", "\\|")
        lines.append(f"| {query['query_id']} | {query['type']} | {query_text} | {target_text} |")
    lines.extend(
        [
            "",
            "## Run against the current backend",
            "",
            "```powershell",
            "python tools/stress_test/run_backend_stress.py --limit 100",
            "python tools/stress_test/evaluate_predictions.py `",
            "  --queries data/stress_test_1200/queries/all_queries.jsonl `",
            "  --predictions data/stress_test_1200/predictions.jsonl",
            "```",
            "",
            "Start with `--limit 100`. A full run can invoke thousands of retrieval calls "
            "and may incur Gemini/API usage when LLM re-ranking is enabled.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def remove_stale_output(output: Path) -> None:
    if not output.exists():
        return
    resolved = output.resolve()
    if resolved.name != "stress_test_1200" or "data" not in resolved.parts:
        raise ValueError(f"Refusing to clean unexpected output path: {resolved}")
    shutil.rmtree(resolved)


def build(args: argparse.Namespace) -> None:
    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    if args.queries_only:
        samples = []
        with (output / "manifest.jsonl").open("r", encoding="utf-8-sig") as stream:
            samples = [json.loads(line) for line in stream if line.strip()]
        textual, qa, trake = generate_queries(samples)
        query_dir = output / "queries"
        write_jsonl(query_dir / "textual_kis.jsonl", textual)
        write_jsonl(query_dir / "qa.jsonl", qa)
        write_jsonl(query_dir / "trake.jsonl", trake)
        write_jsonl(query_dir / "all_queries.jsonl", [*textual, *qa, *trake])
        write_markdown(
            output / "STRESS_TEST_QUERIES.md",
            samples=samples,
            textual=textual,
            qa=qa,
            trake=trake,
        )
        summary_path = output / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["queries"] = {
            "textual_kis": len(textual),
            "qa": len(qa),
            "trake": len(trake),
            "total": len(textual) + len(qa) + len(trake),
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        log(json.dumps(summary["queries"], ensure_ascii=False, indent=2))
        return
    if args.fresh:
        remove_stale_output(output)
    output.mkdir(parents=True, exist_ok=True)
    scratch_root = output / ".range_parts"

    quotas = allocate_quotas(args.total)
    samples: list[dict[str, Any]] = []
    started = time.perf_counter()
    for archive_index, spec in enumerate(ARCHIVES, start=1):
        quota = quotas[spec.key]
        log(
            f"[{archive_index:02d}/{len(ARCHIVES)}] {spec.filename}: "
            f"selecting {quota} frames"
        )
        with RemoteZip(spec.url, timeout=args.timeout) as archive:
            grouped = image_members(archive)
            selected = select_members(grouped, quota)
            log(
                f"  archive videos={len(grouped)}, members={sum(map(len, grouped.values()))}, "
                f"selected videos={len({parse_image_identity(info)[0] for info in selected})}"
            )
            member_destinations = []
            for info in selected:
                video_id, keyframe_index, extension = parse_image_identity(info)
                relative_image = Path("frames") / video_id / f"{keyframe_index:04d}.{extension}"
                member_destinations.append((info, output / relative_image))
            batch_extract_remote_members(
                spec.url,
                member_destinations,
                scratch_root=scratch_root,
                timeout=args.timeout,
                parallel=args.parallel,
                batch_size=args.transfer_batch_size,
            )

            frame_maps: dict[str, dict[int, dict[str, Any]]] = {}
            media_cache: dict[str, dict[str, Any]] = {}
            for selected_index, info in enumerate(selected, start=1):
                video_id, keyframe_index, extension = parse_image_identity(info)
                relative_image = Path("frames") / video_id / f"{keyframe_index:04d}.{extension}"
                destination = output / relative_image
                data = destination.read_bytes()
                validate_image_bytes(data, destination.suffix)

                frame_map = frame_maps.setdefault(
                    video_id, load_frame_map(repo_root, video_id)
                )
                map_item = frame_map.get(keyframe_index, {})
                source_frame_index = int(map_item.get("frame_idx", keyframe_index))
                media = media_cache.setdefault(video_id, load_media_info(repo_root, video_id))
                sample = {
                    "sample_id": f"sample-{len(samples) + 1:04d}",
                    "dataset": spec.dataset,
                    "archive_part": spec.part or None,
                    "video_id": video_id,
                    "keyframe_index": keyframe_index,
                    "source_frame_index": source_frame_index,
                    "timestamp_seconds": map_item.get("pts_time"),
                    "fps": map_item.get("fps"),
                    "retrieval_frame_id": f"{video_id}_f{keyframe_index:04d}",
                    "submission_frame_id": source_frame_index,
                    "relative_image_path": relative_image.as_posix(),
                    "image_sha256": sha256_bytes(data),
                    "image_bytes": len(data),
                    "source_keyframe_archive": spec.url,
                    "source_keyframe_member": info.filename,
                    "title": media.get("title", ""),
                    "author": media.get("author", ""),
                    "publish_date": media.get("publish_date"),
                    "watch_url": media.get("watch_url"),
                    "video_keywords": video_keywords(media),
                }
                samples.append(sample)
                if selected_index % 20 == 0 or selected_index == len(selected):
                    log(f"  downloaded/resumed {selected_index}/{len(selected)}")

    log(f"Enriching {len(samples)} frames from {OBJECTS_URL}")
    with RemoteZip(OBJECTS_URL, timeout=args.timeout) as object_archive:
        object_index = build_object_member_index(object_archive)
        log(f"  object members indexed: {len(object_index)}")
        object_members = []
        for sample in samples:
            key = (sample["video_id"], sample["keyframe_index"])
            info = object_index.get(key)
            if info is None:
                continue
            relative_object = (
                Path("objects")
                / sample["video_id"]
                / f"{sample['keyframe_index']:04d}.json"
            )
            object_members.append((info, output / relative_object))
        batch_extract_remote_members(
            OBJECTS_URL,
            object_members,
            scratch_root=scratch_root,
            timeout=args.timeout,
            parallel=args.parallel,
            batch_size=args.transfer_batch_size,
        )
        for index, sample in enumerate(samples, start=1):
            key = (sample["video_id"], sample["keyframe_index"])
            info = object_index.get(key)
            raw: Any = {}
            if info is not None:
                relative_object = (
                    Path("objects")
                    / sample["video_id"]
                    / f"{sample['keyframe_index']:04d}.json"
                )
                destination = output / relative_object
                object_bytes = destination.read_bytes()
                try:
                    raw = json.loads(object_bytes.decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raw = {}
                sample["relative_object_path"] = relative_object.as_posix()
                sample["source_object_archive"] = OBJECTS_URL
                sample["source_object_member"] = info.filename
            else:
                sample["relative_object_path"] = None
                sample["source_object_archive"] = OBJECTS_URL
                sample["source_object_member"] = None

            detections, counts, labels = summarise_detections(
                raw, args.object_threshold
            )
            sample["object_threshold"] = args.object_threshold
            sample["detections"] = detections
            sample["object_counts"] = counts
            sample["object_keywords"] = labels[:12]
            sample["keywords"] = dedupe_phrases(
                [*sample["object_keywords"], *sample["video_keywords"]], 20
            )
            if index % 100 == 0 or index == len(samples):
                log(f"  object enrichment {index}/{len(samples)}")

    samples.sort(
        key=lambda item: (
            item["dataset"], item["video_id"], item["keyframe_index"]
        )
    )
    for index, sample in enumerate(samples, start=1):
        sample["sample_id"] = f"sample-{index:04d}"

    textual, qa, trake = generate_queries(samples)
    manifest_count = write_jsonl(output / "manifest.jsonl", samples)
    write_manifest_csv(output / "manifest.csv", samples)
    query_dir = output / "queries"
    write_jsonl(query_dir / "textual_kis.jsonl", textual)
    write_jsonl(query_dir / "qa.jsonl", qa)
    write_jsonl(query_dir / "trake.jsonl", trake)
    all_queries = [*textual, *qa, *trake]
    write_jsonl(query_dir / "all_queries.jsonl", all_queries)
    write_markdown(
        output / "STRESS_TEST_QUERIES.md",
        samples=samples,
        textual=textual,
        qa=qa,
        trake=trake,
    )

    summary = {
        "schema_version": 1,
        "frames": manifest_count,
        "videos": len({sample["video_id"] for sample in samples}),
        "datasets": dict(sorted(Counter(sample["dataset"] for sample in samples).items())),
        "queries": {
            "textual_kis": len(textual),
            "qa": len(qa),
            "trake": len(trake),
            "total": len(all_queries),
        },
        "object_threshold": args.object_threshold,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "notes": [
            "Pseudo-ground truth; not manually annotated.",
            "Official submission frame indices come from data/map-keyframes.",
            "Valid ranges use +/-5 source frames for engineering evaluation.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    log(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root / "data" / "stress_test_1200",
    )
    parser.add_argument("--total", type=int, default=DEFAULT_TOTAL)
    parser.add_argument(
        "--object-threshold", type=float, default=DEFAULT_OBJECT_THRESHOLD
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--parallel",
        type=int,
        default=12,
        help="Maximum simultaneous curl range transfers.",
    )
    parser.add_argument(
        "--transfer-batch-size",
        type=int,
        default=50,
        help="Ranges submitted per curl process (kept below Windows command limits).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete and rebuild only data/stress_test_1200 before starting.",
    )
    parser.add_argument(
        "--queries-only",
        action="store_true",
        help="Regenerate query files from an existing manifest without network access.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.total <= 0:
        raise SystemExit("--total must be positive")
    if not 0.0 <= args.object_threshold <= 1.0:
        raise SystemExit("--object-threshold must be between 0 and 1")
    if args.parallel < 1 or args.transfer_batch_size < 1:
        raise SystemExit("--parallel and --transfer-batch-size must be positive")
    try:
        build(args)
    except KeyboardInterrupt:
        log("Interrupted. Re-run the same command to resume existing downloads.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
