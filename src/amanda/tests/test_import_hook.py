import concurrent.futures

import amanda


def test_is_enabled():
    assert amanda.is_enabled()
    with amanda.disabled():
        assert not amanda.is_enabled()
        with amanda.enabled():
            assert amanda.is_enabled()
    with amanda.enabled():
        assert amanda.is_enabled()
        with amanda.disabled():
            assert not amanda.is_enabled()


def test_is_enabled_multi_thread():
    def another_thread():
        assert amanda.is_enabled()
        with amanda.disabled():
            assert not amanda.is_enabled()

    assert amanda.is_enabled()
    with amanda.disabled():
        assert not amanda.is_enabled()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(another_thread)
            concurrent.futures.wait([future])
            future.result()


def another_process():
    assert not amanda.is_enabled()
    with amanda.enabled():
        assert amanda.is_enabled()


def test_is_enabled_multi_process():
    assert amanda.is_enabled()
    with amanda.disabled():
        assert not amanda.is_enabled()
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(another_process)
            concurrent.futures.wait([future])
            future.result()
