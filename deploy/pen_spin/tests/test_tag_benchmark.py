# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Wuji Technology Co., Ltd.
"""Independent scoring, annotation validity, and lossless paired recording."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from lib.printed_tag_template import paper_tags, project_paper
from lib.tag_dataset import PairDataset, PairRecorder, load_truth
from lib.tag_evidence import EvidenceDetection as TagDetection
from lib.tag_metrics import latency_summary, score_frame, summarize_scores


def tag(tid=0, offset=0, instance='one'):
    return dict(instance=instance, family='PEN', id=tid,
                corners=[[offset, 0], [offset+20, 0], [offset+20, 20], [offset, 20]])


def detection(tid=0, offset=0, source='decoded'):
    return TagDetection('PEN', tid, np.array(tag(tid, offset)['corners'], float), source=source)


def test_predictions_never_inflate_recognition():
    from lib.stereo_tag_preview import SheetMissStats
    predicted = detection(source='tracked_only')
    assert SheetMissStats('pen', True).update([predicted], [])['views']['left']['miss_current'] == 108
    result = score_frame([detection(source='tracked_only')], [tag()])
    assert (result['tp'], result['fp'], result['fn']) == (0, 0, 1)
    assert result['predictions'] == result['prediction_hits'] == 1
    candidate = detection(source='decoded_corrected')
    candidate.confirmed = False
    result = score_frame([candidate], [tag()])
    assert result['tp'] == 0 and result['fn'] == 1 and result['tentative'] == 1


def test_wrong_id_duplicate_id_and_background_false_positives():
    truth = [tag(0, 0, 'first'), tag(0, 40, 'second')]
    result = score_frame([detection(0, 0), detection(0, 0), detection(1, 40), detection(0, 100)], truth)
    assert (result['tp'], result['fp'], result['fn'], result['wrong_id']) == (1, 3, 1, 1)
    negative = score_frame([detection()], [])
    summary = summarize_scores([result, negative])
    assert summary['negative_view_frames'] == 1
    assert summary['negative_frame_fp_rate'] == 1
    assert summary['false_discovery_rate'] == 4/5
    assert summary['recall'] == .5


def test_recording_is_lossless_complete_and_ground_truth_is_not_inferred(tmp_path):
    folder = tmp_path/'clip'
    left = np.arange(80*80, dtype=np.uint8).reshape(80, 80)
    writer = PairRecorder(folder, dict(dataset_id='test', kind='real', scene='positive'))
    writer.append(left, 255-left, dict(frame_number=10, host_receive_ns=100))
    writer.close()
    dataset = PairDataset(folder)
    a, b = dataset.images(dataset.records[0])
    np.testing.assert_array_equal(a, left)
    np.testing.assert_array_equal(b, 255-left)
    path = tmp_path/'truth.json'
    truth = dict(schema_version=1, dataset_id='test', source='manual', frames={})
    path.write_text(json.dumps(truth))
    with pytest.raises(ValueError, match='No reviewed'):
        load_truth(dataset, path)
    truth['frames']['0'] = {side: dict(reviewed=True, tags=[]) for side in ('left', 'right')}
    path.write_text(json.dumps(truth))
    with pytest.raises(ValueError, match='explicitly confirmed'):
        load_truth(dataset, path)
    for view in truth['frames']['0'].values():
        view['negative_confirmed'] = True
    path.write_text(json.dumps(truth))
    assert load_truth(dataset, path)['frames']['0']['left']['tags'] == []
    with pytest.raises(FileExistsError):
        PairRecorder(folder, {})


def test_partial_recordings_are_not_evaluation_data(tmp_path):
    writer = PairRecorder(tmp_path/'partial', dict(dataset_id='p', kind='real'))
    writer.close(complete=False)
    with pytest.raises(ValueError, match='incomplete'):
        PairDataset(tmp_path/'partial')


def test_analytic_templates_have_unique_instances_and_canonical_geometry():
    for sheet, count in (('pen', 108), ('wrist', 12)):
        tags = paper_tags(sheet)
        assert len(tags) == count
        assert len({t['instance'] for t in tags}) == count
        mapped = project_paper(sheet, [[0, 0], [420, 0], [420, 594], [0, 594]], (600, 500))
        for original, projected in zip(tags, mapped, strict=True):
            np.testing.assert_allclose(projected['corners'], original['corners']*2, atol=1e-4)
            assert projected['visible']
    assert latency_summary([]) is None
    summary = latency_summary([1, 2, 3, 40])
    assert summary['over_33ms_fraction'] == .25


def test_cli_replay_always_compares_baseline_on_identical_independent_truth(tmp_path):
    from lib.tag_fixture import generate_fixture
    path = Path(__file__).parents[1]/'tools/tag_benchmark.py'
    spec = importlib.util.spec_from_file_location('benchmark_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    truth = generate_fixture(tmp_path/'fixture', frames=5)
    args = SimpleNamespace(dataset=tmp_path/'fixture', truth=truth, output=tmp_path/'report',
                           profiles=['ecc1'], warmup=1)
    module.evaluate(args)
    result = json.loads((tmp_path/'report/comparison.json').read_text())
    assert set(result) == {'baseline', 'ecc1'}
    assert result['ecc1']['real_50cm_validated'] is False
    assert (tmp_path/'report/comparison.txt').is_file()
    base = json.loads((tmp_path/'report/baseline.json').read_text())
    assert base['views']['left']['negative_view_frames'] == 1
    assert base['skipped_capture_frames'] == 0


def test_wrong_direction_and_duplicate_labels_cannot_hide_errors():
    quad = np.asarray(tag()['corners'], float)
    wrong_direction = TagDetection('PEN', 0, np.roll(quad, 1, axis=0))
    score = score_frame([wrong_direction], [tag()])
    assert score['tp'] == 0 and score['fp'] == score['fn'] == score['wrong_orientation'] == 1
    score = score_frame([detection(1), detection(0)], [tag()])
    assert score['tp'] == 1 and score['fp'] == 1 and score['fn'] == 0
