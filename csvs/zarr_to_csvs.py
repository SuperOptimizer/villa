#!/usr/bin/env python3
"""Convert Zarr / OME-Zarr / CSVS volumes to CSVS format.

Usage:
    # Single level
    python zarr_to_csvs.py input.zarr output.csvs

    # Convert all levels of an OME-Zarr pyramid
    python zarr_to_csvs.py input.zarr output.csvs --pyramid

    # Convert OME-Zarr but only levels 0-3
    python zarr_to_csvs.py input.zarr output.csvs --pyramid --levels 4

    # Single array, generate downsampled pyramid from it
    python zarr_to_csvs.py input.zarr/0 output.csvs --pyramid --levels 6

    # Re-encode a CSVS volume with a different codec
    python zarr_to_csvs.py input.csvs output.csvs --pyramid --codec av1
"""

import argparse
import ctypes
import json
import math
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr

# ── ctypes bindings to libcsvs ──────────────────────────────────────────────

class CsvsShardMap(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
        ("fd", ctypes.c_int),
        ("key", ctypes.c_uint64),
    ]

class CsvsVolume(ctypes.Structure):
    _fields_ = [
        ("path", ctypes.c_char * 1024),
        ("shape", ctypes.c_size_t * 3),
        ("padded_shape", ctypes.c_size_t * 3),
        ("chunk_size", ctypes.c_uint32),
        ("shard_size", ctypes.c_uint32),
        ("chunks_per_shard", ctypes.c_uint32),
        ("codec", ctypes.c_int),
        ("codec_level", ctypes.c_int),
        ("shard_cache", ctypes.POINTER(CsvsShardMap)),
    ]

CODEC_MAP = {"lz4": 0, "zstd": 1, "h264": 2, "h265": 3, "av1": 4}


def _load_libcsvs():
    here = Path(__file__).resolve().parent
    lib_path = here / "libcsvs.so"
    if not lib_path.exists():
        raise RuntimeError(
            f"libcsvs.so not found at {lib_path}. "
            "Build with cmake --build build, then cp build/libcsvs.so ."
        )
    lib = ctypes.CDLL(str(lib_path))

    lib.csvs_create.restype = ctypes.c_int
    lib.csvs_create.argtypes = [
        ctypes.POINTER(CsvsVolume), ctypes.c_char_p, ctypes.c_size_t * 3,
        ctypes.c_uint32, ctypes.c_uint32,
        ctypes.c_int, ctypes.c_int,
    ]
    lib.csvs_open.restype = ctypes.c_int
    lib.csvs_open.argtypes = [ctypes.POINTER(CsvsVolume), ctypes.c_char_p]
    lib.csvs_write_shard.restype = ctypes.c_int
    lib.csvs_write_shard.argtypes = [
        ctypes.POINTER(CsvsVolume), ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
    ]
    lib.csvs_read_shard.restype = ctypes.c_int
    lib.csvs_read_shard.argtypes = [
        ctypes.POINTER(CsvsVolume), ctypes.c_size_t, ctypes.c_size_t,
        ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p,
    ]
    lib.csvs_close.restype = None
    lib.csvs_close.argtypes = [ctypes.POINTER(CsvsVolume)]
    return lib


# ── Per-process cache (for multiprocessing workers) ──────────────────────────

_cache_lib = None
_cache_zarr = {}      # zarr_path -> zarr array
_cache_vol = {}       # csvs_path -> CsvsVolume (output)
_cache_src_vol = {}   # csvs_path -> CsvsVolume (input)


def _get_lib():
    global _cache_lib
    if _cache_lib is None:
        _cache_lib = _load_libcsvs()
    return _cache_lib


def _get_zarr(zarr_path):
    if zarr_path not in _cache_zarr:
        _cache_zarr[zarr_path] = zarr.open(zarr_path, mode="r")
    return _cache_zarr[zarr_path]


def _get_src_vol(csvs_path):
    if csvs_path not in _cache_src_vol:
        lib = _get_lib()
        vol = CsvsVolume()
        rc = lib.csvs_open(ctypes.byref(vol), csvs_path.encode())
        if rc != 0:
            raise RuntimeError(f"csvs_open failed for {csvs_path}")
        _cache_src_vol[csvs_path] = vol
    return _cache_src_vol[csvs_path]


def _get_vol(csvs_path):
    if csvs_path not in _cache_vol:
        lib = _get_lib()
        vol = CsvsVolume()
        rc = lib.csvs_open(ctypes.byref(vol), csvs_path.encode())
        if rc != 0:
            raise RuntimeError(f"csvs_open failed for {csvs_path}")
        _cache_vol[csvs_path] = vol
    return _cache_vol[csvs_path]


# ── Downsampling ─────────────────────────────────────────────────────────────

def _downsample_2x(block):
    z, y, x = block.shape
    return block.reshape(z // 2, 2, y // 2, 2, x // 2, 2).mean(axis=(1, 3, 5))


def _downsample_block(block, factor, dtype):
    steps = int(round(math.log2(factor)))
    for _ in range(steps):
        z, y, x = block.shape
        z, y, x = z & ~1, y & ~1, x & ~1
        if z == 0 or y == 0 or x == 0:
            break
        block = _downsample_2x(block[:z, :y, :x])
    return block.astype(dtype)


# ── Worker function (runs in subprocess) ─────────────────────────────────────

def _shard_exists(csvs_path, sz, sy, sx):
    return os.path.exists(os.path.join(csvs_path, "shards",
                                       f"{sz}_{sy}_{sx}.shard"))


def _shard_worker(task):
    """Process one shard. Runs in a worker process.

    task = (src_path, csvs_path, sz, sy, sx, downsample_factor, src_type, overwrite)
    src_type = "zarr", "csvs", or "csvs_downsample"
    downsample_factor=1 means direct copy, >1 means read factor× region and
    block-average down.
    """
    src_path, csvs_path, sz, sy, sx, ds_factor, src_type, overwrite = task

    if not overwrite and _shard_exists(csvs_path, sz, sy, sx):
        return

    lib = _get_lib()
    vol = _get_vol(csvs_path)

    shard_size = vol.shard_size
    chunk_size = vol.chunk_size
    cps = vol.chunks_per_shard
    n_chunks = cps * cps * cps
    np_dtype = np.dtype("uint8")
    raw_chunk = chunk_size ** 3

    if src_type == "csvs":
        # read shard directly from source csvs
        src_vol = _get_src_vol(src_path)
        chunks_arr = np.zeros(n_chunks * raw_chunk, dtype=np.uint8)
        mask = np.zeros(n_chunks, dtype=np.uint8)
        rc = lib.csvs_read_shard(
            ctypes.byref(src_vol),
            ctypes.c_size_t(sz), ctypes.c_size_t(sy), ctypes.c_size_t(sx),
            chunks_arr.ctypes.data_as(ctypes.c_void_p),
            mask.ctypes.data_as(ctypes.c_void_p),
        )
        if rc != 0:
            raise RuntimeError(f"csvs_read_shard failed ({sz},{sy},{sx})")

        rc = lib.csvs_write_shard(
            ctypes.byref(vol),
            ctypes.c_size_t(sz), ctypes.c_size_t(sy), ctypes.c_size_t(sx),
            chunks_arr.ctypes.data_as(ctypes.c_void_p),
            mask.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(raw_chunk),
        )
        if rc != 0:
            raise RuntimeError(f"csvs_write_shard failed ({sz},{sy},{sx})")
        return

    if src_type == "csvs_downsample":
        # read 2x2x2 shards from source csvs, downsample, write one shard
        src_vol = _get_src_vol(src_path)
        src_shard_size = src_vol.shard_size
        src_chunk_size = src_vol.chunk_size
        src_cps = src_vol.chunks_per_shard
        src_n_chunks = src_cps * src_cps * src_cps
        src_raw_chunk = src_chunk_size ** 3

        # Read a 2×shard_size region from the source by reading up to 2x2x2
        # source shards and reconstructing the voxel block
        region = np.zeros((shard_size * 2, shard_size * 2, shard_size * 2),
                          dtype=np.uint8)
        src_chunks = np.zeros(src_n_chunks * src_raw_chunk, dtype=np.uint8)
        src_mask = np.zeros(src_n_chunks, dtype=np.uint8)

        for dz in range(2):
            for dy in range(2):
                for dx in range(2):
                    ssz = sz * 2 + dz
                    ssy = sy * 2 + dy
                    ssx = sx * 2 + dx
                    rc = lib.csvs_read_shard(
                        ctypes.byref(src_vol),
                        ctypes.c_size_t(ssz), ctypes.c_size_t(ssy),
                        ctypes.c_size_t(ssx),
                        src_chunks.ctypes.data_as(ctypes.c_void_p),
                        src_mask.ctypes.data_as(ctypes.c_void_p),
                    )
                    if rc != 0:
                        continue
                    # Unpack chunks into voxel block
                    cdata = src_chunks.reshape(
                        src_cps, src_cps, src_cps,
                        src_chunk_size, src_chunk_size, src_chunk_size)
                    block = (cdata.transpose(0, 3, 1, 4, 2, 5)
                             .reshape(src_shard_size, src_shard_size,
                                      src_shard_size))
                    oz = dz * src_shard_size
                    oy = dy * src_shard_size
                    ox = dx * src_shard_size
                    region[oz:oz + src_shard_size,
                           oy:oy + src_shard_size,
                           ox:ox + src_shard_size] = block

        # Downsample 2x
        ds = _downsample_2x(region).astype(np.uint8)

        # Pad if needed
        if ds.shape != (shard_size, shard_size, shard_size):
            padded = np.zeros((shard_size, shard_size, shard_size),
                              dtype=np.uint8)
            rz = min(ds.shape[0], shard_size)
            ry = min(ds.shape[1], shard_size)
            rx = min(ds.shape[2], shard_size)
            padded[:rz, :ry, :rx] = ds[:rz, :ry, :rx]
            ds = padded

        # Split into chunks and write
        chunks_arr = (ds
                      .reshape(cps, chunk_size, cps, chunk_size,
                               cps, chunk_size)
                      .transpose(0, 2, 4, 1, 3, 5)
                      .reshape(n_chunks, chunk_size, chunk_size, chunk_size))
        chunks_arr = np.ascontiguousarray(chunks_arr)
        mask = np.ones(n_chunks, dtype=np.uint8)

        rc = lib.csvs_write_shard(
            ctypes.byref(vol),
            ctypes.c_size_t(sz), ctypes.c_size_t(sy), ctypes.c_size_t(sx),
            chunks_arr.ctypes.data_as(ctypes.c_void_p),
            mask.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_size_t(raw_chunk),
        )
        if rc != 0:
            raise RuntimeError(f"csvs_write_shard failed ({sz},{sy},{sx})")
        return

    # zarr source path
    arr = _get_zarr(src_path)

    # source read region
    if ds_factor <= 1:
        z0 = sz * shard_size
        y0 = sy * shard_size
        x0 = sx * shard_size
        region = np.array(arr[z0:z0 + shard_size,
                              y0:y0 + shard_size,
                              x0:x0 + shard_size])
    else:
        z0 = sz * shard_size * ds_factor
        y0 = sy * shard_size * ds_factor
        x0 = sx * shard_size * ds_factor
        src_z, src_y, src_x = arr.shape
        iz1 = min(z0 + shard_size * ds_factor, src_z)
        iy1 = min(y0 + shard_size * ds_factor, src_y)
        ix1 = min(x0 + shard_size * ds_factor, src_x)
        src_block = np.array(arr[z0:iz1, y0:iy1, x0:ix1])
        region = _downsample_block(src_block, ds_factor, np_dtype)

    # pad to full shard
    if region.shape != (shard_size, shard_size, shard_size):
        padded = np.zeros((shard_size, shard_size, shard_size), dtype=np_dtype)
        rz = min(region.shape[0], shard_size)
        ry = min(region.shape[1], shard_size)
        rx = min(region.shape[2], shard_size)
        padded[:rz, :ry, :rx] = region[:rz, :ry, :rx]
        region = padded

    # split into chunks (z-major order matching csvs shard layout)
    chunks_arr = (region
                  .reshape(cps, chunk_size, cps, chunk_size, cps, chunk_size)
                  .transpose(0, 2, 4, 1, 3, 5)
                  .reshape(n_chunks, chunk_size, chunk_size, chunk_size))
    chunks_arr = np.ascontiguousarray(chunks_arr)
    mask = np.ones(n_chunks, dtype=np.uint8)

    rc = lib.csvs_write_shard(
        ctypes.byref(vol),
        ctypes.c_size_t(sz), ctypes.c_size_t(sy), ctypes.c_size_t(sx),
        chunks_arr.ctypes.data_as(ctypes.c_void_p),
        mask.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(raw_chunk),
    )
    if rc != 0:
        raise RuntimeError(f"csvs_write_shard failed ({sz},{sy},{sx})")


# ── Run shards across process pool ──────────────────────────────────────────

def _run_shards(tasks, total, workers, prefix=""):
    """Dispatch shard tasks to a ProcessPoolExecutor."""
    if workers <= 1:
        done = 0
        for t in tasks:
            _shard_worker(t)
            done += 1
            print(f"\r{prefix}{done}/{total} shards", end="", flush=True)
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_shard_worker, t): t for t in tasks}
            for f in as_completed(futs):
                f.result()  # propagate exceptions
                done += 1
                print(f"\r{prefix}{done}/{total} shards", end="", flush=True)
    print()


# ── Create csvs volume (main process only) ──────────────────────────────────

def _create_volume(lib, csvs_path, shape, chunk_size, shard_size,
                   codec, codec_level):
    """Call csvs_create in the main process to set up dirs + meta.ini."""
    vol = CsvsVolume()
    shape_arr = (ctypes.c_size_t * 3)(*shape)
    rc = lib.csvs_create(
        ctypes.byref(vol), csvs_path.encode(), shape_arr,
        chunk_size, shard_size,
        CODEC_MAP[codec], codec_level,
    )
    if rc != 0:
        raise RuntimeError(f"csvs_create failed for {csvs_path}")
    lib.csvs_close(ctypes.byref(vol))
    return vol


# ── Convert one level (direct from zarr) ────────────────────────────────────

def _convert_level(lib, src_path, csvs_path, shape,
                   chunk_size, shard_size, codec, codec_level, workers,
                   label="", src_type="zarr", overwrite=False):
    prefix = f"[{label}] " if label else ""
    sz, sy, sx = shape
    print(f"{prefix}shape=({sz}, {sy}, {sx}), src={src_type}")

    _create_volume(lib, csvs_path, shape, chunk_size, shard_size,
                   codec, codec_level)

    n_shards = [(s + shard_size - 1) // shard_size for s in shape]
    total = n_shards[0] * n_shards[1] * n_shards[2]
    print(f"{prefix}shards: {n_shards[0]}x{n_shards[1]}x{n_shards[2]} = {total}")

    tasks = [
        (src_path, csvs_path, sz, sy, sx, 1, src_type, overwrite)
        for sz in range(n_shards[0])
        for sy in range(n_shards[1])
        for sx in range(n_shards[2])
    ]
    _run_shards(tasks, total, workers, prefix)
    return shape


# ── Generate one downsampled level ──────────────────────────────────────────

def _generate_level(lib, source_zarr_path, csvs_path, source_shape, factor,
                    chunk_size, shard_size, codec, codec_level,
                    workers, label="", overwrite=False):
    tgt = tuple(s // factor for s in source_shape)
    if any(d == 0 for d in tgt):
        print(f"[{label}] skipped — too small after {factor}x downsample")
        return None

    prefix = f"[{label}] " if label else ""
    print(f"{prefix}shape={tgt}, downsample={factor}x")

    _create_volume(lib, csvs_path, tgt, chunk_size, shard_size,
                   codec, codec_level)

    n_shards = [(s + shard_size - 1) // shard_size for s in tgt]
    total = n_shards[0] * n_shards[1] * n_shards[2]
    print(f"{prefix}shards: {n_shards[0]}x{n_shards[1]}x{n_shards[2]} = {total}")

    tasks = [
        (source_zarr_path, csvs_path, sz, sy, sx, factor, "zarr", overwrite)
        for sz in range(n_shards[0])
        for sy in range(n_shards[1])
        for sx in range(n_shards[2])
    ]
    _run_shards(tasks, total, workers, prefix)
    return tgt


# ── Root meta.ini (for VC volume loading) ────────────────────────────────────

def _write_root_meta_ini(zarr_path, csvs_path, levels_info):
    """Generate a root meta.ini for the csvs volume.

    If the zarr source has a meta.json (VC volume metadata), copy relevant
    fields.  Otherwise synthesize from the level-0 shape.
    """
    # Try to find meta.json in the zarr source (may be a group root)
    meta = {}
    for candidate in [zarr_path, os.path.dirname(zarr_path)]:
        meta_json = os.path.join(candidate, "meta.json")
        if os.path.isfile(meta_json):
            with open(meta_json) as f:
                meta = json.load(f)
            break

    shape = levels_info[0]["shape"] if levels_info else (0, 0, 0)

    lines = [
        f"format=csvs",
    ]
    lines.append(f"uuid={meta.get('uuid', os.path.basename(csvs_path))}")
    lines.append(f"name={meta.get('name', os.path.basename(csvs_path))}")
    lines.append(f"shape={shape[0]},{shape[1]},{shape[2]}")
    if "voxelsize" in meta:
        lines.append(f"voxelsize={meta['voxelsize']}")
    lines.append(f"min={meta.get('min', 0)}")
    lines.append(f"max={meta.get('max', 0)}")

    ini_path = os.path.join(csvs_path, "meta.ini")
    with open(ini_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {ini_path}")


# ── OME metadata ─────────────────────────────────────────────────────────────

def _write_ome_metadata(output_dir, levels):
    datasets = []
    for i, info in enumerate(levels):
        scale = float(2 ** i)
        datasets.append({
            "path": info["path"],
            "shape": list(info["shape"]),
            "coordinateTransformations": [
                {"type": "scale", "scale": [scale, scale, scale]}
            ],
        })
    meta = {
        "multiscales": [{
            "version": "0.4",
            "axes": [
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": datasets,
        }]
    }
    path = os.path.join(output_dir, "multiscales.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {path}")


# ── Detect input type ────────────────────────────────────────────────────────

def _is_csvs_input(path):
    """Check if path is a CSVS volume (has meta.ini with format=csvs)."""
    meta = os.path.join(path, "meta.ini")
    if not os.path.isfile(meta):
        return False
    with open(meta) as f:
        for line in f:
            if line.strip().startswith("format=csvs"):
                return True
    return False


def _open_csvs_input(csvs_path):
    """Open a CSVS pyramid input. Returns list of (level_key, sub_path, shape)."""
    # Check for sub-level directories (0/, 1/, ...)
    entries = sorted([e for e in os.listdir(csvs_path)
                      if e.isdigit() and os.path.isdir(os.path.join(csvs_path, e))],
                     key=int)
    if entries:
        levels = []
        for key in entries:
            sub = os.path.join(csvs_path, key)
            if _is_csvs_input(sub):
                lib = _load_libcsvs()
                vol = CsvsVolume()
                rc = lib.csvs_open(ctypes.byref(vol), sub.encode())
                if rc != 0:
                    raise RuntimeError(f"csvs_open failed for {sub}")
                shape = tuple(vol.shape[i] for i in range(3))
                chunk_size = vol.chunk_size
                shard_size = vol.shard_size
                lib.csvs_close(ctypes.byref(vol))
                levels.append({"key": key, "path": sub, "shape": shape,
                               "chunk_size": chunk_size, "shard_size": shard_size})
        return levels
    # Single level csvs
    if _is_csvs_input(csvs_path):
        lib = _load_libcsvs()
        vol = CsvsVolume()
        rc = lib.csvs_open(ctypes.byref(vol), csvs_path.encode())
        if rc != 0:
            raise RuntimeError(f"csvs_open failed for {csvs_path}")
        shape = tuple(vol.shape[i] for i in range(3))
        chunk_size = vol.chunk_size
        shard_size = vol.shard_size
        lib.csvs_close(ctypes.byref(vol))
        return [{"key": "0", "path": csvs_path, "shape": shape,
                 "chunk_size": chunk_size, "shard_size": shard_size}]
    raise ValueError(f"Not a valid CSVS volume: {csvs_path}")


def _open_zarr_input(zarr_path):
    """Returns (group_or_none, array_or_none, level_keys)."""
    z = zarr.open(zarr_path, mode="r")
    if isinstance(z, zarr.Group):
        level_keys = sorted([k for k in z.keys() if k.isdigit()], key=int)
        if level_keys:
            return z, None, level_keys
        for key in ("0", "data", "s0"):
            if key in z and hasattr(z[key], "shape"):
                return None, z[key], []
        keys = list(z.keys())
        if len(keys) == 1 and hasattr(z[keys[0]], "shape"):
            return None, z[keys[0]], []
        raise ValueError(f"Cannot find array in group, keys: {keys}")
    else:
        return None, z, []


# ── Entry points ─────────────────────────────────────────────────────────────

def convert_single(input_path, csvs_path, chunk_size, shard_size,
                   codec, codec_level, workers, overwrite=False):
    lib = _load_libcsvs()

    if _is_csvs_input(input_path):
        levels = _open_csvs_input(input_path)
        lvl = levels[0]
        _convert_level(lib, lvl["path"], csvs_path, lvl["shape"],
                       chunk_size, shard_size, codec, codec_level, workers,
                       src_type="csvs", overwrite=overwrite)
        print(f"Done: {csvs_path}")
        return

    group, arr, level_keys = _open_zarr_input(input_path)
    if arr is None:
        arr = group[level_keys[0]]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D, got {arr.ndim}D shape {arr.shape}")

    arr_path = input_path
    if group is not None:
        arr_path = os.path.join(input_path, level_keys[0])

    _convert_level(lib, arr_path, csvs_path, arr.shape,
                   chunk_size, shard_size, codec, codec_level, workers,
                   overwrite=overwrite)
    print(f"Done: {csvs_path}")


def convert_pyramid(input_path, csvs_path, chunk_size, shard_size,
                    codec, codec_level, workers, num_levels=None,
                    level_offset=0, overwrite=False):
    lib = _load_libcsvs()
    os.makedirs(csvs_path, exist_ok=True)
    levels_info = []

    if _is_csvs_input(input_path) or (
            os.path.isdir(input_path) and
            any(_is_csvs_input(os.path.join(input_path, e))
                for e in os.listdir(input_path) if e.isdigit())):
        # ── CSVS input ──
        src_levels = _open_csvs_input(input_path)

        # Apply level offset: input level N+offset → output level N
        if level_offset > 0:
            if level_offset >= len(src_levels):
                raise ValueError(
                    f"--level-offset {level_offset} >= {len(src_levels)} "
                    f"available levels")
            src_levels = src_levels[level_offset:]
            print(f"Level offset {level_offset}: mapping input levels "
                  f"{level_offset}-{level_offset + len(src_levels) - 1} "
                  f"→ output 0-{len(src_levels) - 1}")

        if num_levels is not None:
            src_levels = src_levels[:num_levels]
        print(f"CSVS pyramid: {len(src_levels)} levels")

        # Copy existing levels
        for out_idx, lvl in enumerate(src_levels):
            out_key = str(out_idx)
            level_csvs = os.path.join(csvs_path, out_key)
            shape = _convert_level(lib, lvl["path"], level_csvs, lvl["shape"],
                                   chunk_size, shard_size, codec,
                                   codec_level, workers,
                                   label=f"level {out_key} (src {lvl['key']})",
                                   src_type="csvs", overwrite=overwrite)
            levels_info.append({"path": out_key, "shape": shape})

        # If we need more levels (num_levels requested > available source
        # levels after offset), generate by downsampling the last copied level
        if num_levels is not None and len(levels_info) < num_levels:
            # Use the last copied level as downsample source via zarr-style
            # read — but we only have csvs. Instead, read from the last
            # output csvs level and downsample.
            last_shape = levels_info[-1]["shape"]
            last_csvs = os.path.join(csvs_path, levels_info[-1]["path"])
            for extra in range(len(levels_info), num_levels):
                ds_shape = tuple(s // 2 for s in last_shape)
                if any(d == 0 for d in ds_shape):
                    print(f"[level {extra}] skipped — too small")
                    break
                out_key = str(extra)
                level_csvs = os.path.join(csvs_path, out_key)
                # Read from previous output level, downsample via zarr worker
                # For csvs→csvs downsample we'd need a new worker mode;
                # for now just copy at half-res by reading region-by-region.
                # Simpler: just create the volume and use csvs_read_region
                # Actually, let's just note we need this generated.
                print(f"[level {extra}] generating {ds_shape} by 2x "
                      f"downsample of level {extra - 1}")
                _create_volume(lib, level_csvs, ds_shape, chunk_size,
                               shard_size, codec, codec_level)
                n_shards = [(s + shard_size - 1) // shard_size
                            for s in ds_shape]
                total = n_shards[0] * n_shards[1] * n_shards[2]
                print(f"[level {extra}] shards: "
                      f"{n_shards[0]}x{n_shards[1]}x{n_shards[2]} = {total}")
                tasks = [
                    (last_csvs, level_csvs, sz, sy, sx, 2, "csvs_downsample", overwrite)
                    for sz in range(n_shards[0])
                    for sy in range(n_shards[1])
                    for sx in range(n_shards[2])
                ]
                _run_shards(tasks, total, workers, f"[level {extra}] ")
                levels_info.append({"path": out_key, "shape": ds_shape})
                last_shape = ds_shape
                last_csvs = level_csvs

        _write_ome_metadata(csvs_path, levels_info)
        _write_root_meta_ini(input_path, csvs_path, levels_info)
        print(f"Done: {csvs_path}  ({len(levels_info)} levels)")
        return

    # ── Zarr input ──
    group, arr, level_keys = _open_zarr_input(input_path)

    if group is not None and level_keys:
        # ── Mode A: existing OME-Zarr with multiple levels ──
        if num_levels is not None:
            level_keys = level_keys[:num_levels]
        print(f"OME-Zarr pyramid: {len(level_keys)} levels {level_keys}")

        for key in level_keys:
            level_arr = group[key]
            if level_arr.ndim != 3:
                print(f"  Skipping level {key}: {level_arr.ndim}D")
                continue
            arr_path = os.path.join(input_path, key)
            level_csvs = os.path.join(csvs_path, key)
            shape = _convert_level(lib, arr_path, level_csvs, level_arr.shape,
                                   chunk_size, shard_size, codec,
                                   codec_level, workers, label=f"level {key}",
                                   overwrite=overwrite)
            levels_info.append({"path": key, "shape": shape})

    else:
        # ── Mode B: single array, generate pyramid from it ──
        if arr is None:
            raise ValueError("No array found")
        if arr.ndim != 3:
            raise ValueError(f"Expected 3D, got {arr.ndim}D shape {arr.shape}")

        if num_levels is None:
            min_dim = min(arr.shape)
            num_levels = 1
            while min_dim // (2 ** num_levels) >= shard_size:
                num_levels += 1
            print(f"Auto-detected {num_levels} pyramid levels")
        else:
            print(f"Generating {num_levels} pyramid levels")

        src_path = input_path

        # level 0
        level_csvs = os.path.join(csvs_path, "0")
        shape = _convert_level(lib, src_path, level_csvs, arr.shape,
                               chunk_size, shard_size, codec, codec_level,
                               workers, label="level 0",
                               overwrite=overwrite)
        levels_info.append({"path": "0", "shape": shape})

        # levels 1..N
        for lvl in range(1, num_levels):
            factor = 2 ** lvl
            level_csvs = os.path.join(csvs_path, str(lvl))
            shape = _generate_level(lib, src_path, level_csvs, arr.shape,
                                    factor, chunk_size, shard_size,
                                    codec, codec_level, workers,
                                    label=f"level {lvl}",
                                    overwrite=overwrite)
            if shape is None:
                break
            levels_info.append({"path": str(lvl), "shape": shape})

    _write_ome_metadata(csvs_path, levels_info)
    _write_root_meta_ini(input_path, csvs_path, levels_info)
    print(f"Done: {csvs_path}  ({len(levels_info)} levels)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Zarr / OME-Zarr / CSVS to CSVS")
    parser.add_argument("input_path", help="Input Zarr, OME-Zarr, or CSVS path")
    parser.add_argument("csvs_path", help="Output CSVS path")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--codec", default="lz4",
                        choices=["lz4", "zstd", "h264", "h265", "av1"])
    parser.add_argument("--codec-level", type=int, default=1)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Parallel worker processes (default: cpu count)")
    parser.add_argument("--pyramid", action="store_true",
                        help="Convert/generate full multi-resolution pyramid")
    parser.add_argument("--levels", type=int, default=None,
                        help="Number of pyramid levels (auto if omitted)")
    parser.add_argument("--level-offset", type=int, default=0,
                        help="Skip first N input levels (CSVS pyramid input). "
                             "Input level N becomes output level 0.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing shards (default: skip)")
    args = parser.parse_args()

    if args.pyramid:
        convert_pyramid(
            args.input_path, args.csvs_path, args.chunk_size, args.shard_size,
            args.codec, args.codec_level, args.workers, args.levels,
            level_offset=args.level_offset, overwrite=args.overwrite,
        )
    else:
        convert_single(
            args.input_path, args.csvs_path, args.chunk_size, args.shard_size,
            args.codec, args.codec_level, args.workers,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    multiprocessing.set_start_method("forkserver", force=True)
    main()
