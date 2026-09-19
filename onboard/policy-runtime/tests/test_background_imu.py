from __future__ import annotations

import threading

import numpy as np
import pytest

from drobot_policy_runtime.sources import BackgroundImuSource, ImuSample


def sample(timestamp: float = 100.0) -> ImuSample:
    return ImuSample(
        np.zeros(3, dtype=np.float32),
        np.array([0.0, 0.0, -1.0], dtype=np.float32),
        np.array([0.0, 0.0, 9.81], dtype=np.float32),
        0.0,
        timestamp,
    )


def test_cached_imu_read_stays_available_while_owner_is_blocked_then_expires():
    now = [100.0]
    second_read = threading.Event()
    release_read = threading.Event()
    closed = threading.Event()
    device_threads = []

    class Device:
        reads = 0

        def __init__(self):
            device_threads.append(threading.get_ident())

        def read(self):
            device_threads.append(threading.get_ident())
            self.reads += 1
            if self.reads > 1:
                second_read.set()
                assert release_read.wait(2.0)
            return sample(now[0])

        def close(self):
            device_threads.append(threading.get_ident())
            closed.set()

    source = BackgroundImuSource(Device, clock=lambda: now[0], close_timeout_s=0.01)
    try:
        source.start()
        assert second_read.wait(1.0)
        # This must use the cached sample; a device call here would block.
        cached = source.read()
        cached.projected_gravity_body[:] = 0.0
        assert source.read().projected_gravity_body[2] == -1.0
        now[0] += 0.121
        with pytest.raises(RuntimeError, match="stale"):
            source.read()
        source.close()
        assert source.cleanup_pending
        assert not closed.is_set()  # Never cross-thread-close an active read.
    finally:
        release_read.set()
        assert closed.wait(1.0)
        source.close()
    assert len(set(device_threads)) == 1
    assert device_threads[0] != threading.get_ident()


def test_background_read_error_is_propagated_even_while_cache_is_fresh():
    fail = threading.Event()
    closed = threading.Event()

    class Device:
        reads = 0

        def read(self):
            self.reads += 1
            if self.reads > 1:
                assert fail.wait(2.0)
                raise OSError("I2C read failed")
            return sample()

        def close(self):
            closed.set()

    source = BackgroundImuSource(Device, clock=lambda: 100.0)
    try:
        source.start()
        fail.set()
        assert closed.wait(1.0)
        with pytest.raises(RuntimeError, match="I2C read failed"):
            source.read()
    finally:
        fail.set()
        source.close()


def test_cancelled_initializer_retains_owner_until_factory_returns():
    factory_entered = threading.Event()
    release_factory = threading.Event()
    closed = threading.Event()
    caller_finished = threading.Event()
    cancel = threading.Event()
    errors = []
    owner_threads = []

    class Device:
        def read(self):
            pytest.fail("Cancelled initialization must not begin acquisition")

        def close(self):
            owner_threads.append(threading.get_ident())
            closed.set()

    def factory():
        owner_threads.append(threading.get_ident())
        factory_entered.set()
        assert release_factory.wait(2.0)
        return Device()

    source = BackgroundImuSource(factory, close_timeout_s=0.01)

    def prepare():
        try:
            source.start(cancel_event=cancel)
        except RuntimeError as exc:
            errors.append(str(exc))
        finally:
            caller_finished.set()

    caller = threading.Thread(target=prepare)
    caller.start()
    try:
        assert factory_entered.wait(1.0)
        cancel.set()
        assert caller_finished.wait(1.0)
        assert errors and "cancelled" in errors[0]
        assert source.cleanup_pending
        assert not closed.is_set()
    finally:
        release_factory.set()
        assert closed.wait(1.0)
        caller.join(timeout=1.0)
        source.close()
    assert len(set(owner_threads)) == 1


def test_start_rejects_incomplete_first_sample_and_closes_device():
    closed = threading.Event()

    class Device:
        def read(self):
            value = sample()
            return ImuSample(np.zeros(2), value.projected_gravity_body,
                             value.linear_acceleration_body_m_s2, 0.0, 100.0)

        def close(self):
            closed.set()

    source = BackgroundImuSource(Device, clock=lambda: 100.0)
    try:
        with pytest.raises(RuntimeError, match="complete finite vectors"):
            source.start()
        assert closed.wait(1.0)
    finally:
        source.close()


def test_startup_timeout_does_not_close_factory_on_callers_thread():
    release_factory = threading.Event()
    closed = threading.Event()

    class Device:
        def read(self):
            pytest.fail("Timed-out session must not start acquisition")

        def close(self):
            closed.set()

    def factory():
        assert release_factory.wait(2.0)
        return Device()

    source = BackgroundImuSource(factory, startup_timeout_s=0.02, close_timeout_s=0.01)
    try:
        with pytest.raises(RuntimeError, match="timed out"):
            source.start()
        assert source.cleanup_pending
        assert not closed.is_set()
    finally:
        release_factory.set()
        assert closed.wait(1.0)
        source.close()
