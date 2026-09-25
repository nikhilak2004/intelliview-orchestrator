from unittest.mock import MagicMock

import pytest

from orchestrator.load_balancer import BalancingStrategy, LoadBalancer


@pytest.fixture
def workers():
    return [
        {
            "worker_id": "worker-1",
            "active_tasks": 0,
            "capacity": 4,
        },
        {
            "worker_id": "worker-2",
            "active_tasks": 1,
            "capacity": 4,
        },
        {
            "worker_id": "worker-3",
            "active_tasks": 2,
            "capacity": 4,
        },
    ]


def make_load_balancer(strategy):
    load_balancer = LoadBalancer.__new__(LoadBalancer)
    load_balancer.worker_registry = MagicMock()
    load_balancer.strategy = strategy
    load_balancer.round_robin_index = 0
    load_balancer._wrr_current_weights = {}
    from threading import Lock

    load_balancer._wrr_lock = Lock()
    return load_balancer


def test_round_robin_selects_workers_in_order(workers):
    load_balancer = make_load_balancer(BalancingStrategy.ROUND_ROBIN)
    load_balancer._get_cached_workers = MagicMock(return_value=workers)

    selected = [load_balancer.select_worker()["worker_id"] for _ in range(4)]

    assert selected == [
        "worker-1",
        "worker-2",
        "worker-3",
        "worker-1",
    ]


def test_round_robin_returns_none_when_no_workers():
    load_balancer = make_load_balancer(BalancingStrategy.ROUND_ROBIN)
    load_balancer._get_cached_workers = MagicMock(return_value=[])

    assert load_balancer.select_worker() is None


def test_least_loaded_selects_worker_with_fewest_tasks(workers):
    load_balancer = make_load_balancer(BalancingStrategy.LEAST_LOADED)
    load_balancer.worker_registry.get_least_loaded_worker.return_value = workers[0]

    selected = load_balancer.select_worker()

    assert selected["worker_id"] == "worker-1"


def test_least_loaded_returns_none_when_no_workers():
    load_balancer = make_load_balancer(BalancingStrategy.LEAST_LOADED)
    load_balancer.worker_registry.get_least_loaded_worker.return_value = None

    assert load_balancer.select_worker() is None


def test_queue_based_returns_worker_when_available(workers):
    load_balancer = make_load_balancer(BalancingStrategy.QUEUE_BASED)
    load_balancer.worker_registry.get_least_loaded_worker.return_value = workers[1]

    selected = load_balancer.select_worker()

    assert selected["worker_id"] == "worker-2"


def test_queue_based_returns_none_when_no_workers():
    load_balancer = make_load_balancer(BalancingStrategy.QUEUE_BASED)
    load_balancer.worker_registry.get_least_loaded_worker.return_value = None

    assert load_balancer.select_worker() is None


def test_weighted_least_loaded_uses_relative_load():
    workers = [
        {
            "worker_id": "worker-1",
            "active_tasks": 3,
            "capacity": 4,
            "weight": 4,
        },
        {
            "worker_id": "worker-2",
            "active_tasks": 2,
            "capacity": 4,
            "weight": 1,
        },
    ]

    load_balancer = make_load_balancer(BalancingStrategy.WEIGHTED_LEAST_LOADED)
    load_balancer.worker_registry.get_available_workers.return_value = workers

    selected = load_balancer.select_worker()

    assert selected["worker_id"] == "worker-1"


def test_weighted_round_robin_returns_none_without_workers():
    load_balancer = make_load_balancer(BalancingStrategy.WEIGHTED_ROUND_ROBIN)
    load_balancer.worker_registry.get_available_workers.return_value = []

    assert load_balancer.select_worker() is None


def test_weighted_round_robin_distributes_by_weight():
    workers = [
        {
            "worker_id": "worker-1",
            "active_tasks": 0,
            "capacity": 4,
            "weight": 2,
        },
        {
            "worker_id": "worker-2",
            "active_tasks": 0,
            "capacity": 4,
            "weight": 1,
        },
    ]

    load_balancer = make_load_balancer(BalancingStrategy.WEIGHTED_ROUND_ROBIN)
    load_balancer.worker_registry.get_available_workers.return_value = workers

    selected = [load_balancer.select_worker()["worker_id"] for _ in range(3)]

    assert selected.count("worker-1") == 2
    assert selected.count("worker-2") == 1


def test_switch_strategy():
    load_balancer = make_load_balancer(BalancingStrategy.ROUND_ROBIN)

    load_balancer.switch_strategy(BalancingStrategy.LEAST_LOADED)

    assert load_balancer.strategy == BalancingStrategy.LEAST_LOADED


import time


def test_get_cached_workers_uses_cache_within_ttl(monkeypatch):
    """Within the TTL window, the registry should only be queried once."""
    workers_v1 = [{"worker_id": "worker-1", "active_tasks": 0, "capacity": 4}]

    load_balancer = LoadBalancer(
        strategy=BalancingStrategy.LEAST_LOADED, worker_registry=MagicMock()
    )
    load_balancer.worker_registry.get_available_workers.return_value = workers_v1

    fake_time = [1000.0]
    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    first = load_balancer._get_cached_workers()
    fake_time[0] += 1  # still well within the default 5s TTL
    second = load_balancer._get_cached_workers()

    assert first == workers_v1
    assert second == workers_v1
    assert load_balancer.worker_registry.get_available_workers.call_count == 1


def test_get_cached_workers_refreshes_after_ttl_expires(monkeypatch):
    """Once the TTL elapses, the cache should be refreshed from the registry,
    reflecting newly available/unavailable workers."""
    workers_v1 = [{"worker_id": "worker-1", "active_tasks": 0, "capacity": 4}]
    workers_v2 = [{"worker_id": "worker-2", "active_tasks": 0, "capacity": 4}]

    load_balancer = LoadBalancer(
        strategy=BalancingStrategy.LEAST_LOADED, worker_registry=MagicMock()
    )
    load_balancer.worker_registry.get_available_workers.side_effect = [
        workers_v1,
        workers_v2,
    ]

    fake_time = [1000.0]
    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    first = load_balancer._get_cached_workers()
    fake_time[0] += load_balancer._cache_ttl + 1  # past the TTL
    second = load_balancer._get_cached_workers()

    assert first == workers_v1
    assert second == workers_v2
    assert load_balancer.worker_registry.get_available_workers.call_count == 2


def test_get_cached_workers_respects_ttl_when_no_workers_available(monkeypatch):
    """Regression test for the duplicate implementation this issue removed:
    one version used `not self._worker_cache`, which treats an empty list as
    'uninitialized' and forces a registry lookup on every call. The TTL must
    still be respected even when zero workers are currently available."""
    load_balancer = LoadBalancer(
        strategy=BalancingStrategy.LEAST_LOADED, worker_registry=MagicMock()
    )
    load_balancer.worker_registry.get_available_workers.return_value = []

    fake_time = [1000.0]
    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    load_balancer._get_cached_workers()
    fake_time[0] += 1
    load_balancer._get_cached_workers()
    fake_time[0] += 1
    load_balancer._get_cached_workers()

    assert load_balancer.worker_registry.get_available_workers.call_count == 1
