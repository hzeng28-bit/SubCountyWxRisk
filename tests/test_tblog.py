import pytest

pytest.importorskip("tensorboard")
torch = pytest.importorskip("torch")

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from wxrisk.tblog import RunLog  # noqa: E402


def _read(path):
    ea = EventAccumulator(str(path), size_guidance={"tensors": 0, "histograms": 0})
    ea.Reload()
    return ea


def test_scalars_text_and_histograms_are_written(tmp_path):
    log = RunLog("g/run", {"lr": 1e-3, "seed": 0}, root=tmp_path)
    assert log.active
    m = torch.nn.Linear(3, 1)
    m(torch.randn(4, 3)).sum().backward()
    for i in range(5):
        log.scalar("train/loss", 1.0 / (i + 1), i)
    log.scalars("epoch/loss", {"train": 1.0, "val": 2.0}, 1)
    log.weights(m, 1)
    log.close()
    ea = _read(tmp_path / "g" / "run")
    assert [e.step for e in ea.Scalars("train/loss")] == [0, 1, 2, 3, 4]
    assert "hparams/text_summary" in ea.Tags()["tensors"]
    assert "weights/weight" in ea.Tags()["histograms"] and "grads/weight" in ea.Tags()["histograms"]


def test_restarted_run_replaces_the_old_directory(tmp_path):
    a = RunLog("g/run", root=tmp_path)
    a.scalar("x", 1, 0)
    a.close()
    b = RunLog("g/run", root=tmp_path)
    b.scalar("y", 1, 0)
    b.close()
    tags = _read(tmp_path / "g" / "run").Tags()["scalars"]
    assert "y" in tags and "x" not in tags


def test_disabled_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("WXRISK_TB", "0")
    log = RunLog("g/run", {"a": 1}, root=tmp_path)
    assert not log.active
    log.scalar("x", 1, 0)
    log.weights(torch.nn.Linear(1, 1), 0)
    log.close()
    assert not (tmp_path / "g").exists()
