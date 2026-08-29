import numpy as np

from experiments.dense_decoder import disk_average_probabilities, select_decoder
from experiments.dense_protocol import full_plan, stage1_candidates, stage2_candidates, stage3_candidates
from experiments.detection_protocol import (
    deim_candidates,
    detr_candidates,
    full_detection_plan,
    siteait_candidates,
)


def test_protocol_run_accounting():
    assert len(stage1_candidates()) == 21
    assert len(stage2_candidates()) == 60
    assert len(stage3_candidates()) == 15
    assert len(full_plan()) == 96


def test_stage1_model_counts():
    counts = {}
    for candidate in stage1_candidates():
        counts[candidate.model] = counts.get(candidate.model, 0) + 1
    assert counts == {
        "unet": 3, "transunet": 2, "setr": 6, "segmenter": 6, "segman_t": 4
    }


def test_detection_search_accounting():
    assert len(detr_candidates()) == 216
    assert len(deim_candidates()) == 216
    assert len(siteait_candidates()) == 12
    assert len(full_detection_plan()) == 444


def test_disk_average_and_nested_decoder_selection():
    maps = np.zeros((4, 9, 9), dtype=float)
    site_xy = np.array([[4.0, 4.0]])
    labels = np.array([[0], [0], [1], [1]], dtype=bool)
    maps[0:2] = 0.1
    maps[2:4] = 0.9
    scores = disk_average_probabilities(maps, site_xy, radius=2)
    assert scores.shape == (4, 1)
    selected = select_decoder(maps, labels, site_xy)
    assert selected.accuracy == 1.0
    assert selected.youden_j == 1.0
    assert 0.30 <= selected.threshold <= 0.70
