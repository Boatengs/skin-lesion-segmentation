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
