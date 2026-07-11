"""Tests for host-level doctor checks (driver hold + log rotation)."""

from __future__ import annotations

from vserve import host_health


class TestNvidiaDriverHeld:
    def test_ok_when_no_apt_mark(self, mocker):
        mocker.patch("vserve.host_health.shutil.which", return_value=None)
        out = host_health.check_nvidia_driver_held()
        assert out["ok"] is True
        assert "skipped" in out["message"]

    def test_ok_when_no_nvidia_packages(self, mocker):
        mocker.patch("vserve.host_health._held_packages", return_value={"foo", "bar"})
        mocker.patch("vserve.host_health._installed_nvidia_packages", return_value=set())
        out = host_health.check_nvidia_driver_held()
        assert out["ok"] is True

    def test_ok_when_nvidia_held(self, mocker):
        mocker.patch("vserve.host_health._held_packages", return_value={"nvidia-driver-open", "other"})
        mocker.patch(
            "vserve.host_health._installed_nvidia_packages",
            return_value={"nvidia-driver-open", "libnvidia-compute-580"},
        )
        out = host_health.check_nvidia_driver_held()
        assert out["ok"] is True
        assert "held (1/2)" in out["message"]

    def test_warns_when_nvidia_unheld(self, mocker):
        mocker.patch("vserve.host_health._held_packages", return_value=set())
        mocker.patch(
            "vserve.host_health._installed_nvidia_packages",
            return_value={"nvidia-driver-open"},
        )
        out = host_health.check_nvidia_driver_held()
        assert out["ok"] is False
        assert "NOT apt-held" in out["message"]
        assert "apt-mark hold" in out["fix"]

    def test_skips_when_apt_mark_fails(self, mocker):
        mocker.patch("vserve.host_health._held_packages", return_value=None)
        out = host_health.check_nvidia_driver_held()
        assert out["ok"] is True


class TestLogRotation:
    def test_ok_when_dir_absent(self, tmp_path):
        out = host_health.check_log_rotation(tmp_path / "nope")
        assert out["ok"] is True

    def test_ok_when_covered_by_rule(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "vllm.log").write_bytes(b"x" * 10)
        rotate_d = tmp_path / "logrotate.d"
        rotate_d.mkdir()
        (rotate_d / "vllm").write_text(f"{log_dir}/*.log {{\n  copytruncate\n}}\n")
        out = host_health.check_log_rotation(log_dir, rotate_d=rotate_d)
        assert out["ok"] is True
        assert "configured" in out["message"]

    def test_ok_when_no_logs_yet(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        rotate_d = tmp_path / "logrotate.d"
        rotate_d.mkdir()
        out = host_health.check_log_rotation(log_dir, rotate_d=rotate_d)
        assert out["ok"] is True

    def test_warns_when_uncovered_and_large(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "vllm.log").write_bytes(b"x" * (3 * 1024 * 1024))
        rotate_d = tmp_path / "logrotate.d"
        rotate_d.mkdir()  # empty — no rule
        out = host_health.check_log_rotation(log_dir, rotate_d=rotate_d)
        assert out["ok"] is False
        assert "No logrotate rule" in out["message"]
        assert "3 MB" in out["message"]
        assert "copytruncate" in out["fix"]

    def test_rule_for_other_dir_does_not_count_as_coverage(self, tmp_path):
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        (log_dir / "vllm.log").write_bytes(b"x" * 10)
        rotate_d = tmp_path / "logrotate.d"
        rotate_d.mkdir()
        # A rule for a DIFFERENT directory must not count as coverage — an
        # uncovered dir that already has logs warns regardless of size.
        (rotate_d / "other").write_text("/var/log/other/*.log { copytruncate }\n")
        out = host_health.check_log_rotation(log_dir, rotate_d=rotate_d)
        assert out["ok"] is False
        assert "configured" not in out["message"]
