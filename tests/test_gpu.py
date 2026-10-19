from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from daft.internal import gpu


def _fake_nvml(init_rc: int = 0, count_rc: int = 0, count: int = 0) -> MagicMock:
    nvml = MagicMock()
    nvml.nvmlInit.return_value = init_rc

    def get_count(ref):
        ref._obj.value = count
        return count_rc

    nvml.nvmlDeviceGetCount_v2.side_effect = get_count
    return nvml


def test_visible_devices_parsed_from_env(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", " 0, ,2 ")
    assert gpu.cuda_visible_devices() == ["0", "2"]


def test_empty_env_means_no_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu.cuda_visible_devices() == []


def test_visible_devices_fall_back_to_nvml_count(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu, "_raw_device_count_nvml", lambda: 2)
    assert gpu.cuda_visible_devices() == ["0", "1"]


def test_nvml_library_missing():
    with patch("ctypes.CDLL", side_effect=OSError("libnvidia-ml.so.1 not found")):
        assert gpu._raw_device_count_nvml() == 0


def test_nvml_reports_device_count():
    fake = _fake_nvml(count=3)
    with patch("ctypes.CDLL", return_value=fake) as cdll:
        assert gpu._raw_device_count_nvml() == 3
    cdll.assert_called_once_with("libnvidia-ml.so.1")


def test_nvml_init_failure_warns():
    with (
        patch("ctypes.CDLL", return_value=_fake_nvml(init_rc=1)),
        pytest.warns(UserWarning, match="initialize NVML"),
    ):
        assert gpu._raw_device_count_nvml() == 0


def test_nvml_device_count_failure_warns():
    with (
        patch("ctypes.CDLL", return_value=_fake_nvml(count_rc=1, count=5)),
        pytest.warns(UserWarning, match="device count"),
    ):
        assert gpu._raw_device_count_nvml() == 0
