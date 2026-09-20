from pathlib import Path

import numpy as np
import pandas as pd

from hydrogeo_insar.extensometer.analysis import _read_marker_table


def test_marker_table_to_interval_compaction(tmp_path: Path):
    frame = pd.DataFrame(
        {
            "标孔": ["深度（m）", "2020/01/15", "2021/01/15"],
            "F1": [41.0, -100.0, -120.0],
            "F2": [150.0, -70.0, -80.0],
            "F3": [267.0, -30.0, -35.0],
            "F4": [401.0, 0.0, 0.0],
        }
    )
    path = tmp_path / "ext.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    sec = {
        "date_column": "标孔",
        "depth_row_label": "深度（m）",
        "marker_columns": ["F1", "F2", "F3", "F4"],
        "marker_positive": "uplift",
    }
    markers, intervals, profile, marker_def = _read_marker_table(path, sec)

    assert [x[1] for x in marker_def] == [41.0, 150.0, 267.0, 401.0]
    first = intervals[intervals["date"] == pd.Timestamp("2020-01-15")]
    assert np.allclose(first["compaction_mm"].to_numpy(), [30.0, 40.0, 30.0])
    assert np.isclose(profile.loc[0, "profile_compaction_mm"], 100.0)
    assert np.isclose(profile.loc[1, "profile_compaction_mm"], 120.0)
    assert len(markers) == 8
