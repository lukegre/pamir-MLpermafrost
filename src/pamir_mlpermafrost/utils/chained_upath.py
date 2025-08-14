"""
chained_upath.py
==================

This module provides a small wrapper around the standard library's
``pathlib`` and Python's built‑in file handling to mimic a subset of
``upath.UPath`` for environments where the ``universal_pathlib`` package is
unavailable.  Its main purpose is to preserve so‑called *fsspec chaining*
semantics for URIs containing ``"::"`` sequences.  The standard
``upath.UPath`` derives its protocol from the first portion of a URL and
normalises redundant slashes; it uses a regular expression that only
matches a single protocol followed by ``://`` or ``:/``【928695604485132†L17-L21】.
That regex does not understand chained protocols like
``"simplecache::s3://bucket/path"`` and will therefore drop one of the
slashes, yielding ``"simplecache::s3:/bucket/path"``.  This class avoids
normalising away the double slash and leaves the chain unchanged in the
string representation.

The implementation focuses on local file paths.  For URIs with a chain
prefix (the text before the first ``"::"``), the chain is preserved but
ignored when reading or writing: the inner URI after the ``"::"`` is
interpreted as either a ``file`` URI or a plain POSIX path.  The class
does not implement network protocols such as ``s3`` or ``http``; those
would normally be provided by ``fsspec`` itself.  Nevertheless, the
public API is intentionally similar to ``UPath`` for common operations
like ``open()``, ``read_text()``, ``write_text()``, and path
manipulations.  This makes it suitable for tests and simple local usage
without depending on external packages.

Key features
------------

* **Chained URIs** – The string representation preserves ``"::"``
  prefixes and leaves ``"//"`` intact after the protocol.  When
  constructing a path, a URI such as ``"simplecache::file://tmp/foo"``
  will round‑trip via ``str()`` without losing the double slash.

* **Path operations** – Attributes like ``name``, ``suffix`` and
  ``stem`` are proxied through a ``pathlib.PurePosixPath`` created from
  the inner path.  ``parent`` and ``joinpath()`` return new
  ``ChainedUPath`` objects with the same chain and protocol.

* **File I/O** – For ``file`` protocols or plain paths, the class
  delegates to Python's built‑in ``open()`` function.  ``read_text()``
  reads the file as UTF‑8, and ``write_text()`` writes a string and
  returns the number of characters written.  Methods like ``exists()``
  use ``Path.exists()`` on the inner filesystem path.

This module is intended for demonstration and testing purposes.  It
should not be relied upon for production use with remote filesystems or
complex fsspec behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator, Optional, Union
from urllib.parse import urlsplit


class ChainedUPath:
    """A minimal path class that preserves fsspec chaining syntax.

    Parameters
    ----------
    uri : str or path‑like
        A string that may contain a protocol (e.g. ``"file://"``) and
        optionally an fsspec chaining prefix separated by ``"::"``.  The
        chain, if present, is preserved exactly in the string
        representation.  Only local file paths are supported for the
        inner protocol.
    *pathsegments : list[str] or path‑like
        Additional path segments are joined to ``uri`` using POSIX rules.
    storage_options : dict, optional
        Ignored in this implementation.  Included for API similarity with
        ``upath.UPath``.

    Examples
    --------
    >>> p = ChainedUPath('simplecache::file://tmp/data.txt')
    >>> str(p)
    'simplecache::file://tmp/data.txt'
    >>> p.name
    'data.txt'
    >>> p.read_text()  # doctest: +SKIP
    'contents of the file'
    """

    def __init__(
        self,
        uri: Union[str, os.PathLike[str]],
        *pathsegments: Union[str, os.PathLike[str]],
        storage_options: Optional[dict] = None,
    ) -> None:
        self._storage_options = storage_options or {}
        # Coerce the first argument to a string
        if hasattr(uri, "__fspath__"):
            base = uri.__fspath__()
        else:
            base = str(uri)
        # Join additional segments onto the base path.  We do this early
        # because the chain prefix applies only to the first part of the
        # path.  ``pathsegments`` are joined using PurePosixPath.
        if pathsegments:
            joined = PurePosixPath(base)
            for seg in pathsegments:
                if hasattr(seg, "__fspath__"):
                    seg = seg.__fspath__()
                joined = joined.joinpath(str(seg))
            base = joined.as_posix()
        self._uri = base
        # Identify chain prefix and inner URI.  A chain is any prefix
        # preceding the first occurrence of ``"::"``.  We split on the first
        # occurrence only.
        if "::" in base:
            self._chain, inner = base.split("::", 1)
        else:
            self._chain, inner = None, base
        # Parse the inner URI.  We detect a protocol by looking for
        # ``scheme://...``.  A single slash after the colon is also
        # considered; the original UPath implementation normalises ``:/`` to
        # ``://`` when the protocol is known【207450931432691†L416-L424】.  Here we keep the
        # path unchanged and only extract the scheme for informational
        # purposes.
        parts = urlsplit(inner)
        # ``parts.scheme`` will be empty if there is no protocol
        self._protocol = parts.scheme
        if self._protocol:
            # Combine netloc and path for file URIs.  For example,
            # ``file://tmp/data`` yields netloc='tmp', path='/data', so
            # combining them gives '/tmp/data'.  If netloc is empty,
            # ``parts.path`` already contains the desired filesystem path.
            if parts.netloc:
                self._path = "/" + parts.netloc + parts.path
            else:
                # Retain the leading slash for absolute paths, remove
                # nothing for relative paths
                self._path = parts.path or "/"
        else:
            # No protocol: treat the entire inner URI as a POSIX path
            self._path = inner
        # Normalise the path using PurePosixPath to remove redundant
        # separators and up‑level references ("..") without touching the
        # chain prefix.  This does not remove leading slashes.
        self._pure = PurePosixPath(self._path)

    # ------------------------------------------------------------------
    # Representation and basic properties
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        """Return the full URI, preserving any chain and double slashes.

        If a chain prefix was provided, it is emitted exactly as supplied.
        If a protocol is present, ``"://"`` precedes the inner path.
        Otherwise the bare path is returned.  Leading slashes on the
        filesystem path are removed when formatting the URI to avoid
        producing ``"file:////..."`` strings.
        """
        if self._protocol:
            # Remove any leading slash for URI formatting.  The path on
            # disk may still begin with ``/`` (absolute path), but URIs
            # should not have ``file:////`` when a netloc was empty.
            path = self._pure.as_posix().lstrip("/")
            proto = f"{self._protocol}://{path}"
        else:
            proto = self._pure.as_posix()
        if self._chain:
            return f"{self._chain}::{proto}"
        return proto

    def __repr__(self) -> str:
        return f"{type(self).__name__}({str(self)!r})"

    # ------------------------------------------------------------------
    # Path component properties
    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        """The final component of the path, excluding any drive/chain."""
        return self._pure.name

    @property
    def suffix(self) -> str:
        """The file extension of the final component, including the dot."""
        return self._pure.suffix

    @property
    def stem(self) -> str:
        """The final component without its suffix."""
        return self._pure.stem

    @property
    def parts(self) -> tuple[str, ...]:
        """Return a tuple of path parts for the inner filesystem path."""
        # For absolute paths the first part is the root ("/"), which is not
        # meaningful in the context of a chained URI.  Drop the root for
        # consistency with relative paths.
        parts = self._pure.parts
        if parts and parts[0] == "/":
            return parts[1:]
        return parts

    @property
    def parent(self) -> "ChainedUPath":
        """Return the parent directory, preserving the chain and protocol."""
        # Delegate to PurePosixPath.parent to compute the new path
        new_pure = self._pure.parent
        # Reassemble a new URI from the chain and protocol
        inner = new_pure.as_posix()
        if self._protocol:
            uri = f"{self._protocol}://{inner.lstrip('/')}"
        else:
            uri = inner
        if self._chain:
            uri = f"{self._chain}::{uri}"
        return ChainedUPath(uri, storage_options=self._storage_options)

    def with_name(self, name: str) -> "ChainedUPath":
        """Return a new path with the file name changed."""
        if "/" in name:
            raise ValueError(f"Invalid name {name!r}")
        new = self._pure.with_name(name)
        return self._replace_path(new)

    def with_suffix(self, suffix: str) -> "ChainedUPath":
        """Return a new path with the file suffix changed."""
        new = self._pure.with_suffix(suffix)
        return self._replace_path(new)

    def joinpath(self, *other: Union[str, os.PathLike[str]]) -> "ChainedUPath":
        """Join one or more path components to this path and return a new object."""
        new_pure = self._pure
        for part in other:
            if hasattr(part, "__fspath__"):
                part = part.__fspath__()
            new_pure = new_pure.joinpath(str(part))
        return self._replace_path(new_pure)

    def _replace_path(self, new_pure: PurePosixPath) -> "ChainedUPath":
        """Internal helper to construct a new object with a different inner path."""
        # Build new URI
        inner = new_pure.as_posix()
        if self._protocol:
            uri = f"{self._protocol}://{inner.lstrip('/')}"
        else:
            uri = inner
        if self._chain:
            uri = f"{self._chain}::{uri}"
        return ChainedUPath(uri, storage_options=self._storage_options)

    # ------------------------------------------------------------------
    # File I/O operations
    # ------------------------------------------------------------------
    def _local_path(self) -> str:
        """Return the local filesystem path for the inner URI.

        Only ``file`` and empty protocols are supported.  Other
        protocols will raise a ``NotImplementedError``.
        """
        if self._protocol and self._protocol != "file":
            raise NotImplementedError(
                f"Unsupported protocol {self._protocol!r} in {self._uri!r}"
            )
        return str(self._pure)

    def open(self, mode: str = "r", *args, **kwargs):
        """Open the path as a file.

        Only local file paths are supported.  ``mode`` can be any valid
        file mode accepted by Python's built‑in ``open``.  Additional
        positional and keyword arguments are passed through.
        """
        path = self._local_path()
        return open(path, mode, *args, **kwargs)

    def read_text(self, encoding: str = "utf-8") -> str:
        """Read the entire contents of the file as text using UTF‑8 by default."""
        with self.open("r", encoding=encoding) as fh:
            return fh.read()

    def write_text(self, data: str, encoding: str = "utf-8") -> int:
        """Write text to the file, creating parent directories as needed.

        Returns the number of characters written.
        """
        # Ensure parent directories exist
        path = self._local_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.open("w", encoding=encoding) as fh:
            return fh.write(data)

    def read_bytes(self) -> bytes:
        """Read the entire contents of the file as bytes."""
        with self.open("rb") as fh:
            return fh.read()

    def write_bytes(self, data: bytes) -> int:
        """Write bytes to the file, creating parent directories as needed."""
        path = self._local_path()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.open("wb") as fh:
            return fh.write(data)

    # ------------------------------------------------------------------
    # Filesystem queries
    # ------------------------------------------------------------------
    def exists(self) -> bool:
        """Return ``True`` if the file or directory exists on the local filesystem."""
        try:
            path = self._local_path()
        except NotImplementedError:
            return False
        return Path(path).exists()

    def is_file(self) -> bool:
        """Return ``True`` if the path exists and is a file."""
        try:
            path = self._local_path()
        except NotImplementedError:
            return False
        return Path(path).is_file()

    def is_dir(self) -> bool:
        """Return ``True`` if the path exists and is a directory."""
        try:
            path = self._local_path()
        except NotImplementedError:
            return False
        return Path(path).is_dir()

    def iterdir(self) -> Iterator["ChainedUPath"]:
        """Yield ``ChainedUPath`` objects for the contents of a directory."""
        path = self._local_path()
        for child in Path(path).iterdir():
            yield self.joinpath(child.name)

    def glob(self, pattern: str) -> Iterator["ChainedUPath"]:
        """Yield all existing files matching the given glob pattern."""
        path = self._local_path()
        for child in Path(path).glob(pattern):
            # Only return objects that actually exist
            yield self.joinpath(child.name)

    def rglob(self, pattern: str) -> Iterator["ChainedUPath"]:
        """Yield all existing files matching the given recursive glob pattern."""
        path = self._local_path()
        for child in Path(path).rglob(pattern):
            yield self.joinpath(child.as_posix().removeprefix(path).lstrip("/"))
