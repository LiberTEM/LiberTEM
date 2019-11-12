from unittest import mock


import pytest


from libertem.io.dataset.cached import CacheStats, CacheItem, OrphanItem


@pytest.fixture
def cs():
    cs = CacheStats(":memory:")
    cs.initialize_schema()
    yield cs
    cs.close()


@pytest.fixture
def ci():
    return CacheItem(
        dataset="deadbeef",
        partition=5,
        size=768,
        path="/tmp/dont_care",
    )


@pytest.fixture
def oi():
    return OrphanItem(
        size=512,
        path="/tmp/dont_care",
    )


def test_cache_stats_starts_connected(cs):
    assert cs._conn is not None


def test_initialize_schema_idempotent(cs):
    cs.initialize_schema()  # already called in fixture, should not raise here


def test_first_miss(cs, ci):
    with mock.patch('time.time', side_effect=lambda: 42):
        cs.record_miss(ci)
    ds_stats = cs.get_stats_for_dataset("deadbeef")
    stats_items = list(ds_stats.items())
    assert len(stats_items) == 1
    si = stats_items[0]
    assert si[0] == 5
    assert si[1]["last_access"] == 42
    assert si[1]["hits"] == 0


def test_first_hits(cs, ci):
    # Unfortunately, someone is calling time.time() under the hood, and doing it
    # in a nondeterministic way. What I would have liked to test here: passing
    # side_effect=[21, 42] and making sure the 21 is overwritten by the 42.
    # Now we don't know if the first or second value survives.
    with mock.patch('time.time', side_effect=lambda: 42):
        cs.record_miss(ci)
        cs.record_hit(ci)
    ds_stats = cs.get_stats_for_dataset("deadbeef")
    stats_items = list(ds_stats.items())
    assert len(stats_items) == 1
    si = stats_items[0]
    assert si[0] == 5
    assert si[1]["last_access"] == 42
    assert si[1]["hits"] == 1


def test_eviction(cs, ci):
    with mock.patch('time.time', side_effect=lambda: 42):
        cs.record_miss(ci)
        cs.record_hit(ci)
        cs.record_eviction(ci)
    ds_stats = cs.get_stats_for_dataset("deadbeef")
    stats_items = list(ds_stats.items())
    assert len(stats_items) == 0


def test_record_orphan(cs, oi):
    cs.maybe_orphan(oi)

    cursor = cs.query("SELECT * FROM stats")
    assert len(cursor.fetchall()) == 0

    orphans = cs.get_orphans()
    assert len(orphans) == 1


def test_hit_after_orphan(cs, ci, oi):
    cs.maybe_orphan(oi)

    orphans = cs.get_orphans()
    assert len(orphans) == 1

    with mock.patch('time.time', side_effect=lambda: 42):
        cs.record_hit(ci)
    ds_stats = cs.get_stats_for_dataset("deadbeef")
    stats_items = list(ds_stats.items())
    assert len(stats_items) == 1
    si = stats_items[0]
    assert si[0] == 5
    assert si[1]["last_access"] == 42
    assert si[1]["hits"] == 1

    orphans = cs.get_orphans()
    assert len(orphans) == 0


def test_miss_after_orphan(cs, ci, oi):
    """
    db/fs out of sync: the orphan was deleted from fs, then miss'd
    """
    cs.maybe_orphan(oi)

    orphans = cs.get_orphans()
    assert len(orphans) == 1

    with mock.patch('time.time', side_effect=lambda: 42):
        cs.record_miss(ci)
    ds_stats = cs.get_stats_for_dataset("deadbeef")
    stats_items = list(ds_stats.items())
    assert len(stats_items) == 1
    si = stats_items[0]
    assert si[0] == 5
    assert si[1]["last_access"] == 42
    assert si[1]["hits"] == 0

    orphans = cs.get_orphans()
    assert len(orphans) == 0
