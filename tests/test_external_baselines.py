import json
import os
import sys

import pytest
import torch
import yaml

from baseline import BASELINES, CATEGORIES, PROCESSING_TYPES
from baseline.external_official import ESAM, MGL, PROVENANCE
from data.dataset_utils import load_dataset
from evaluation.evaluator import Evaluator


def test_esam_mgl_are_official_external_wrappers():
    assert BASELINES['esam'] is ESAM
    assert BASELINES['mgl'] is MGL
    assert CATEGORIES['esam'] == 'external-official'
    assert CATEGORIES['mgl'] == 'external-official'
    assert PROCESSING_TYPES['esam'] == 'external-wrapper'
    assert PROCESSING_TYPES['mgl'] == 'external-wrapper'
    assert 'Non-Displayed Items' in PROVENANCE['esam'].paper_title
    assert 'Meta Graph Learning' in PROVENANCE['mgl'].paper_title


def test_external_wrapper_requires_repo_path():
    with pytest.raises(RuntimeError, match='repo_path'):
        ESAM({}).validate_repo()


def test_external_wrapper_exports_split_and_collects_metrics(tmp_path):
    repo = tmp_path / 'mgl_repo'
    repo.mkdir()
    for relative in PROVENANCE['mgl'].required_files:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('placeholder')
    command = [
        sys.executable,
        '-c',
        "import json, os; out=os.environ['DUAL2FAIR_EXTERNAL_OUTPUT_DIR']; "
        "json.dump({'NDCG':0.1,'Hit':0.2,'DUF':0.3,'DIF':0.4,'UIF':0.5}, "
        "open(os.path.join(out,'test_metrics.json'),'w'))",
    ]
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    evaluator = Evaluator(dataset, split='test', require_uif_reference=False)
    wrapper = MGL({
        'repo_path': str(repo),
        'command': command,
        'output_root': str(tmp_path / 'runs'),
    })
    result = wrapper.run(dataset, evaluator, evaluator)
    assert result['NDCG'] == 0.1
    assert result['external_method'] == 'mgl'
    prepared = wrapper.prepared_dir
    assert os.path.exists(os.path.join(prepared, 'train.csv'))
    assert os.path.exists(os.path.join(prepared, 'validation.csv'))
    assert os.path.exists(os.path.join(prepared, 'test.csv'))
    with open(os.path.join(prepared, 'metadata.json')) as handle:
        metadata = json.load(handle)
    assert metadata['official_repo'] == PROVENANCE['mgl'].repo_url
    assert metadata['split_hash'] == dataset.split_hash


def test_default_config_contains_external_provenance():
    with open('config/default.yaml') as handle:
        settings = yaml.safe_load(handle)
    assert settings['baseline']['esam']['official_repo'] == PROVENANCE['esam'].repo_url
    assert settings['baseline']['mgl']['official_repo'] == PROVENANCE['mgl'].repo_url
