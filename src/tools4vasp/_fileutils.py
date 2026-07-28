#!/usr/bin/env python3
#
# Shared file-handling helpers for tools4vasp
#
import os


def iter_lines_reversed(f, chunk_size=64 * 1024):
    """Yield the lines of a binary file from last to first.

    Reads the file in chunks starting from the end, so only as much of
    the file is touched as the consumer actually iterates over. Useful
    for pulling the final entries out of huge OUTCAR files without
    reading the whole file.

    Input Parameters
    ----------------
    f : binary file object
        Seekable file opened in binary mode

    chunk_size : int
        Number of bytes to read per backwards step, must be positive

    Returns
    -------
    Generator of lines (bytes, without trailing newline), last line first.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be positive, got {chunk_size}")
    f.seek(0, os.SEEK_END)
    pos = f.tell()
    carry = b""
    while pos > 0:
        size = min(chunk_size, pos)
        pos -= size
        f.seek(pos)
        lines = (f.read(size) + carry).split(b"\n")
        # the first element may be a partial line completed by the next
        # (earlier) chunk — hold it back as carry
        carry = lines[0]
        yield from reversed(lines[1:])
    if carry:
        yield carry
