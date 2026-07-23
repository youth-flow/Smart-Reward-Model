from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

import smart_reward.paths as path_module
from smart_reward.paths import relative_posix_reference

ROOT = Path(__file__).parents[1]


def test_relative_posix_reference_relativizes_absolute_paths(tmp_path: Path) -> None:
    base = (tmp_path / "runs" / "seed-1").resolve()
    target = (tmp_path / "artifacts" / "seed-1").resolve()

    reference = relative_posix_reference(target, base=base)

    assert PurePosixPath(reference).parts == ("..", "..", "artifacts", "seed-1")
    assert "\\" not in reference
    assert not Path(reference).is_absolute()


def test_relative_posix_reference_fails_closed_when_relpath_is_impossible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def cross_drive_failure(*_: object, **__: object) -> str:
        raise ValueError("path is on a different drive")

    monkeypatch.setattr(path_module.os.path, "relpath", cross_drive_failure)

    with pytest.raises(ValueError, match="across filesystem drives"):
        relative_posix_reference(tmp_path / "target", base=tmp_path / "base")


def test_persisted_hpc_reports_do_not_dump_absolute_runtime_paths() -> None:
    host_probe = (ROOT / "scripts" / "hpc4" / "host_gpu_probe.sbatch").read_text(encoding="utf-8")
    gpu_smoke = (ROOT / "scripts" / "hpc4" / "gpu_smoke.sbatch").read_text(encoding="utf-8")
    controlled = (ROOT / "scripts" / "hpc4" / "controlled.sbatch").read_text(encoding="utf-8")

    combined_reports = host_probe + gpu_smoke
    for forbidden in (
        'echo "project_root=${project_root}"',
        'echo "submit_dir=${SLURM_SUBMIT_DIR}"',
        'echo "image=${PRORM_IMAGE}"',
        "env | grep '^SLURM_'",
    ):
        assert forbidden not in combined_reports
    assert "SLURM_SUBMIT_DIR" not in controlled
