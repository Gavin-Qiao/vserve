"""Pure helpers for HuggingFace model download flow.

These helpers compute paths, predicates, and filesystem layouts without
any console / picker interaction. The interactive orchestrator
(`_download_model` in cli.py) calls into these to do the data-only work.

Extracted from `cli.py` in 0.6.3 per audit
`docs/audits/2026-05-20-cli-sprawl.md` — the download flow accreted ~410
lines of mixed pure-data + interactive logic.
"""

from __future__ import annotations

import pathlib
import re
import shutil

from vserve.model_files import (
    is_gguf_name,
    iter_recursive_files_with_suffix,
    iter_recursive_weight_files,
    iter_top_level_files_with_suffix,
)


def root_has_top_level_weights(path: pathlib.Path) -> bool:
    return (
        bool(iter_top_level_files_with_suffix(path, ".safetensors"))
        or bool(iter_top_level_files_with_suffix(path, ".bin"))
        or bool(iter_top_level_files_with_suffix(path, ".gguf"))
    )


def variant_common_prefix(files: list[str]) -> str | None:
    """If every file in `files` shares the same first path segment, return it."""
    split_paths = [pathlib.PurePosixPath(name).parts for name in files]
    if not split_paths or not all(len(parts) > 1 for parts in split_paths):
        return None
    prefix = split_paths[0][0]
    if not all(parts[0] == prefix for parts in split_paths):
        return None
    return prefix


def safe_variant_label(label: str) -> str:
    """Slugify a variant label for use in a directory name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-") or "variant"


def gguf_variant_root(
    base_dir: pathlib.Path, *, model_name: str, variant
) -> pathlib.Path:
    """Sibling directory where a single GGUF variant should land as its own
    runnable model root."""
    label = safe_variant_label(str(getattr(variant, "label", "variant")))
    suffix = label if label.lower() not in model_name.lower() else "variant"
    return base_dir.parent / f"{model_name}-{suffix}"


def variant_contains_gguf(variant) -> bool:
    return any(is_gguf_name(filename) for filename in getattr(variant, "files", {}))


def expected_download_roots(
    local_dir: pathlib.Path,
    *,
    model_name: str,
    selected_variants: list,
    is_gguf_download: bool,
) -> list[pathlib.Path]:
    """The directories where weight files will end up after a download.

    For GGUF downloads each variant materializes into its own sibling root
    (because vserve treats each GGUF quant as a distinct ``ModelInfo``).
    For non-GGUF downloads the whole repo lands at the original
    ``local_dir``.
    """
    if is_gguf_download:
        return [
            gguf_variant_root(local_dir, model_name=model_name, variant=variant)
            for variant in selected_variants
        ]
    return [local_dir]


def download_roots_ready(roots: list[pathlib.Path]) -> bool:
    """True when every expected root exists and contains at least one weight file."""
    if not roots:
        return False
    for root in roots:
        try:
            if not root.exists() or not iter_recursive_weight_files(root):
                return False
        except OSError:
            return False
    return True


def strip_downloaded_file_prefix(
    downloaded: pathlib.Path, root: pathlib.Path, filename: str
) -> pathlib.Path:
    """Move a downloaded repo subpath to the runnable root using only its basename.

    Used after a download to flatten subdirectory-organised variants
    (e.g. ``Q4_K_XL/foo.gguf`` → ``foo.gguf`` at the root). Cleans up
    empty parent dirs left behind.
    """
    dest = root / pathlib.PurePosixPath(filename).name
    if downloaded.resolve(strict=False) == dest.resolve(strict=False):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    shutil.move(str(downloaded), str(dest))
    parent = downloaded.parent
    while parent != root and root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent
    return dest


def clear_stale_gguf_files(root: pathlib.Path) -> None:
    """Best-effort delete every .gguf under `root`. Used to drop a leftover
    variant before re-downloading. Silent on errors — caller should not
    depend on a particular file being gone."""
    if not root.exists():
        return
    for stale in iter_recursive_files_with_suffix(root, ".gguf"):
        try:
            stale.unlink()
        except OSError:
            pass


def materialize_subdirectory_variants(
    local_dir: pathlib.Path,
    *,
    model_name: str,
    selected_variants: list,
    shared: dict[str, int],
) -> list[pathlib.Path]:
    """Expose each selected subdirectory HF variant as a separate runnable model root.

    HuggingFace repos sometimes organise multiple quantisations as
    sibling subdirectories (``Q4_K_M/``, ``Q5_K_S/``, ...). vserve expects
    one runnable model per directory, so for each variant we either
    symlink (preferred) or copy the files into a sibling root named
    ``{model_name}-{label}``.

    Returns the list of runnable roots so callers can show them to the user
    or pass them to the limits-cache invalidator.
    """
    if not selected_variants:
        return [local_dir]
    if len(selected_variants) == 1 and root_has_top_level_weights(local_dir):
        ignore = local_dir / ".vserve-ignore"
        if ignore.exists():
            ignore.unlink()
        return [local_dir]

    def link_or_copy(src: pathlib.Path, dest: pathlib.Path) -> None:
        if not src.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            if dest.is_symlink() or dest.exists():
                dest.unlink()
        except OSError:
            return
        try:
            dest.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dest)

    roots: list[pathlib.Path] = []
    materialized_any = False
    materialize_top_level = len(selected_variants) > 1
    for variant in selected_variants:
        variant_files = list(getattr(variant, "files", {}).keys())
        prefix = variant_common_prefix(variant_files)
        safe_label = (
            re.sub(r"[^A-Za-z0-9._-]+", "-", str(getattr(variant, "label", prefix or "variant"))).strip("-")
            or (prefix or "variant")
        )
        if prefix is None and not materialize_top_level:
            roots.append(local_dir)
            continue

        materialized = local_dir.parent / f"{model_name}-{safe_label}"
        materialized.mkdir(parents=True, exist_ok=True)
        materialized_any = True

        for name in shared:
            rel = pathlib.PurePosixPath(name)
            dest_rel = (
                pathlib.PurePosixPath(*rel.parts[1:])
                if prefix is not None and rel.parts and rel.parts[0] == prefix and len(rel.parts) > 1
                else rel
            )
            link_or_copy(local_dir.joinpath(*rel.parts), materialized.joinpath(*dest_rel.parts))

        for name in variant_files:
            rel = pathlib.PurePosixPath(name)
            stripped = pathlib.PurePosixPath(*rel.parts[1:]) if prefix is not None else rel
            link_or_copy(local_dir.joinpath(*rel.parts), materialized.joinpath(*stripped.parts))

        roots.append(materialized)

    ignore = local_dir / ".vserve-ignore"
    if materialized_any and (materialize_top_level or not root_has_top_level_weights(local_dir)):
        ignore.write_text("materialized variants live in sibling model roots\n")
    elif ignore.exists() and root_has_top_level_weights(local_dir):
        ignore.unlink()
    return roots or [local_dir]
