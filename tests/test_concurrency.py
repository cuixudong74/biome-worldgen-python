"""Concurrent generation and recomputation.

* the same chunk is never cross-written (immutable swap under lock);
* reads always observe a complete version;
* parallel requests converge to the same result as a sequential run.
"""

from __future__ import annotations

import threading

from biome_worldgen import World, WorldGenConfig


def _cfg(**kw) -> WorldGenConfig:
    base = dict(chunk_size=4, max_propagation_rounds=300, convergence_epsilon=1e-12,
                worker_threads=4)
    base.update(kw)
    return WorldGenConfig(**base)


def _fingerprint(world: World) -> dict:
    out = {}
    for key in sorted(world.chunk_keys()):
        chunk = world.get_chunk(*key)
        out[key] = tuple(
            (c.height, c.temperature, c.moisture, c.vegetation) for c in chunk.cells
        )
    return out


def test_parallel_disjoint_regions():
    world = World(seed=1414, config=_cfg())
    errors: list[BaseException] = []

    def worker(x0, z0, x1, z1):
        try:
            world.generate_region(x0, z0, x1, z1)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(0, 0, 2, 2)),
        threading.Thread(target=worker, args=(3, 0, 5, 2)),
        threading.Thread(target=worker, args=(0, 3, 2, 5)),
        threading.Thread(target=worker, args=(3, 3, 5, 5)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors

    ref = World(seed=1414, config=_cfg())
    ref.generate_region(0, 0, 5, 5)
    assert _fingerprint(world) == _fingerprint(ref)


def test_same_chunk_concurrent_generation():
    world = World(seed=1515, config=_cfg())
    errors: list[BaseException] = []
    barrier = threading.Barrier(6)

    def worker():
        try:
            barrier.wait()
            world.generate_region(0, 0, 2, 2)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors

    ref = World(seed=1515, config=_cfg())
    ref.generate_region(0, 0, 2, 2)
    assert _fingerprint(world) == _fingerprint(ref)


def test_reads_are_complete_versions():
    """A reader must always observe a complete chunk (never a half update)."""
    world = World(seed=1616, config=_cfg())
    stop = threading.Event()
    torn: list[str] = []

    def reader():
        while not stop.is_set():
            for key in world.chunk_keys():
                chunk = world.get_chunk(*key)
                if chunk is None:
                    continue
                # Every cell must be a fully populated Cell with 4 fields.
                for cell in chunk.cells:
                    if not isinstance(cell.moisture, float):
                        torn.append("bad cell")
                        return
                # Chunk size must be consistent.
                if len(chunk.cells) != chunk.size * chunk.size:
                    torn.append("bad size")
                    return

    def writer():
        for i in range(6):
            world.generate_region(0, 0, 3, 3)

    r = threading.Thread(target=reader)
    w = threading.Thread(target=writer)
    r.start()
    w.start()
    w.join()
    stop.set()
    r.join()
    assert not torn, torn


def test_concurrent_recompute_disjoint():
    world = World(seed=1717, config=_cfg())
    world.generate_region(0, 0, 5, 5)
    errors: list[BaseException] = []

    def worker(key):
        try:
            world.recompute({key: __import__("biome_worldgen").RegionParams(
                temperature_offset=2.0)})
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    keys = [(1, 1), (4, 1), (1, 4), (4, 4)]
    threads = [threading.Thread(target=worker, args=(k,)) for k in keys]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors

    # Reference: full rebuild with all four offsets applied.
    from biome_worldgen import RegionParams
    ref = World(seed=1717, config=_cfg())
    ref.generate_region(0, 0, 5, 5)
    ref.recompute({k: RegionParams(temperature_offset=2.0) for k in keys})
    assert _fingerprint(world) == _fingerprint(ref)
