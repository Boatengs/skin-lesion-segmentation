import sys

from skinseg import report, train


def test_train_and_report(data_dir, tmp_path, monkeypatch):
    for split in ("image", "lesion"):
        info = train.run(split, 42, data_dir, tmp_path / "runs" / f"{split}-s42", epochs=1, base=4)
        assert 0 <= info["test_dice_at_0.5"] <= 1
    monkeypatch.setattr(sys, "argv", ["report", "--runs", str(tmp_path / "runs"), "--data", str(data_dir),
                                      "--out", str(tmp_path / "results")])
    report.main()
    md = (tmp_path / "results" / "RESULTS.md").read_text()
    assert "Every run" in md and (tmp_path / "results" / "examples.png").exists()


def test_resume_matches_uninterrupted(data_dir, tmp_path):
    full = train.run("lesion", 7, data_dir, tmp_path / "full", epochs=3, base=4)
    assert train.run("lesion", 7, data_dir, tmp_path / "part", epochs=3, base=4, budget_minutes=1e-9) is None
    assert (tmp_path / "part" / "state.pt").exists()
    resumed = train.run("lesion", 7, data_dir, tmp_path / "part", epochs=3, base=4)
    assert not (tmp_path / "part" / "state.pt").exists()
    for a, b in zip(full["history"], resumed["history"], strict=True):
        assert abs(a["train_loss"] - b["train_loss"]) < 1e-5 and abs(a["val_loss"] - b["val_loss"]) < 1e-5
