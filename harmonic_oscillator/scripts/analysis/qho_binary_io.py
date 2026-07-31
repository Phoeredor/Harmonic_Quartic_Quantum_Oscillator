#!/usr/bin/env python3
# Decode fixed-layout binary time series of Euclidean-path observables while
# preserving the lattice and run metadata needed by statistical analyses.
"""
Read fixed-size binary measurement files produced by the QHO PIMC executable.

The C code writes one binary file as

    fixed-size header  +  fixed-size measurement records.

The header layout is defined in `include/qho_binary_io.h`; this module mirrors
that layout on the Python side. The record dtype below must therefore remain
synchronized with `qho_binary_record_t` in the C implementation.

The module provides two access patterns:

    iter_records(...)
        streaming chunk reader for large files;

    memmap_records(...)
        NumPy memory map for random access without loading the full file.

When executed as a script, it prints the decoded header, the number of records,
the first and last records, and optionally streaming means of the stored
observables.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


MAGIC = b"QHOPIMC\0"
VERSION = 1
HEADER_SIZE = 256
ENDIAN_MARKER = 0x01020304

RECORD_DTYPE = np.dtype([
    ("sweep", "<i8"),
    ("y_mean", "<f8"),
    ("y2_mean", "<f8"),
    ("dy2_mean", "<f8"),
    ("energy_ren", "<f8"),
    ("acc_rate", "<f8"),
])


def is_binary_qho(path: str | Path) -> bool:
    """Return True if the file starts with the QHO binary magic string."""
    with Path(path).open("rb") as handle:
        return handle.read(len(MAGIC)) == MAGIC


def read_header(path: str | Path) -> dict[str, int | float]:
    """
    Decode and check the fixed-size binary header.

    The function verifies the magic string, format version, header size, record
    size, and endian marker before returning the metadata dictionary.

    Raises
    ------
    ValueError
        If the file is too short or if the binary layout is not supported by
        this reader.
    """
    path = Path(path)

    with path.open("rb") as handle:
        raw = handle.read(HEADER_SIZE)

    if len(raw) != HEADER_SIZE:
        raise ValueError(f"{path}: short binary header")

    if raw[:len(MAGIC)] != MAGIC:
        raise ValueError(f"{path}: bad QHO binary magic")

    offset = len(MAGIC)

    def unpack(fmt: str) -> int | float:
        nonlocal offset
        size = struct.calcsize(fmt)
        value = struct.unpack_from(fmt, raw, offset)[0]
        offset += size
        return value

    header = {
        "version": unpack("<I"),
        "header_size": unpack("<I"),
        "record_size": unpack("<I"),
        "endian_marker": unpack("<I"),
        "nt": unpack("<i"),
        "beta": unpack("<d"),
        "eta": unpack("<d"),
        "n_therm": unpack("<q"),
        "n_sweeps": unpack("<q"),
        "meas_stride": unpack("<q"),
        "seed": unpack("<Q"),
        "stream": unpack("<Q"),
        "update_code": unpack("<i"),
        "init_code": unpack("<i"),
        "n_over": unpack("<i"),
        "observable_mask": unpack("<I"),
        "n_records_expected": unpack("<Q"),
    }

    if header["version"] != VERSION:
        raise ValueError(f"{path}: unsupported binary version {header['version']}")

    if header["header_size"] != HEADER_SIZE:
        raise ValueError(f"{path}: unsupported header size {header['header_size']}")

    if header["record_size"] != RECORD_DTYPE.itemsize:
        raise ValueError(f"{path}: unsupported record size {header['record_size']}")

    if header["endian_marker"] != ENDIAN_MARKER:
        raise ValueError(f"{path}: unsupported endian marker {header['endian_marker']!r}")

    return header


def record_count(path: str | Path, header: dict[str, int | float] | None = None) -> int:
    """
    Return the number of complete records stored after the binary header.

    The file size must be exactly

        header_size + record_count * record_size.

    A mismatch with `n_records_expected` is reported to stderr because it may
    indicate an interrupted run or a file copied before writing completed.
    """
    path = Path(path)
    header = read_header(path) if header is None else header

    size = path.stat().st_size
    payload = size - int(header["header_size"])
    record_size = int(header["record_size"])

    if payload < 0 or payload % record_size != 0:
        raise ValueError(f"{path}: file size is not consistent with fixed record size")

    count = payload // record_size
    expected = int(header["n_records_expected"])

    if count != expected:
        print(
            f"warning: {path}: actual records {count} differ from expected {expected}",
            file=sys.stderr,
        )

    return int(count)


def iter_records(path: str | Path, chunk_records: int = 1_000_000) -> Iterator[np.ndarray]:
    """
    Yield measurement records in chunks.

    This is the preferred access pattern for very large production files,
    because it does not load the full time series into memory.

    Parameters
    ----------
    path:
        Binary QHO measurement file.
    chunk_records:
        Maximum number of records yielded per chunk.

    Yields
    ------
    numpy.ndarray
        Structured array with dtype `RECORD_DTYPE`.
    """
    if chunk_records <= 0:
        raise ValueError("chunk_records must be positive")

    path = Path(path)
    header = read_header(path)
    count = record_count(path, header)

    with path.open("rb") as handle:
        handle.seek(int(header["header_size"]))

        remaining = count
        while remaining > 0:
            n_read = min(chunk_records, remaining)
            data = np.fromfile(handle, dtype=RECORD_DTYPE, count=n_read)

            if data.size != n_read:
                raise OSError(f"{path}: short read while reading records")

            remaining -= int(data.size)
            yield data


def memmap_records(path: str | Path) -> np.memmap:
    """
    Return a read-only memory map of all measurement records.

    The returned object behaves like a NumPy structured array and is useful for
    random access to large files. For simple reductions over all records,
    `iter_records` is usually more memory predictable.
    """
    path = Path(path)
    header = read_header(path)
    count = record_count(path, header)

    return np.memmap(
        path,
        dtype=RECORD_DTYPE,
        mode="r",
        offset=int(header["header_size"]),
        shape=(count,),
    )


def streaming_means(path: str | Path) -> dict[str, float]:
    """
    Compute observable means using streaming chunks.

    The sweep index is not averaged. All other record fields are averaged over
    saved measurements.
    """
    sums = {name: 0.0 for name in RECORD_DTYPE.names if name != "sweep"}
    count = 0

    for chunk in iter_records(path):
        count += int(chunk.size)
        for name in sums:
            sums[name] += float(np.sum(chunk[name]))

    if count == 0:
        raise ValueError(f"{path}: cannot compute means of an empty measurement file")

    return {name: value / count for name, value in sums.items()}


def main() -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="binary QHO measurement file")
    parser.add_argument("--means", action="store_true", help="compute streaming means")
    args = parser.parse_args()

    header = read_header(args.path)
    count = record_count(args.path, header)

    print("header:")
    for key in sorted(header):
        print(f"  {key}: {header[key]}")

    print(f"records: {count}")

    if count:
        records = memmap_records(args.path)
        print("first:", {name: records[0][name].item() for name in RECORD_DTYPE.names})
        print("last:", {name: records[count - 1][name].item() for name in RECORD_DTYPE.names})

    if args.means:
        print("means:")
        for key, value in streaming_means(args.path).items():
            print(f"  {key}: {value:.17g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
