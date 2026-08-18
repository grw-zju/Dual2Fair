import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ExternalBaselineProvenance:
    name: str
    source_title: str
    venue: str
    doi: str
    repo_url: str
    default_branch: str
    required_files: tuple
    default_entrypoint: str


PROVENANCE = {
    'esam': ExternalBaselineProvenance(
        name='ESAM',
        source_title='ESAM: Discriminative Domain Adaptation with Non-Displayed Items to Improve Long-Tail Performance',
        venue='SIGIR 2020',
        doi='10.1145/3397271.3401043',
        repo_url='https://github.com/A-bone1/ESAM',
        default_branch='master',
        required_files=('test_cikm.py', 'models/item_ranking/NeuMF_cikm_p.py',
                        'utils/load_data/load_data_my.py'),
        default_entrypoint='test_cikm.py'),
    'mgl': ExternalBaselineProvenance(
        name='MGL',
        source_title='Meta Graph Learning for Long-tail Recommendation',
        venue='KDD 2023',
        doi='10.1145/3580305.3599428',
        repo_url='https://github.com/weicy15/MGL',
        default_branch='main',
        required_files=('train.py', 'model.py', 'load_data.py', 'metric.py'),
        default_entrypoint='train.py'),
}


class ExternalScoreMatrixModel:
    def __init__(self, scores):
        self.scores = torch.as_tensor(scores, dtype=torch.float32)

    def forward(self, user_ids, item_ids):
        return self.scores[user_ids.cpu(), item_ids.cpu()]


class OfficialExternalBaseline:
    method_key = None

    def __init__(self, config=None):
        self.config = config or {}
        self.provenance = PROVENANCE[self.method_key]
        self.repo_commit = None
        self.prepared_dir = None
        self._best_validation_results = None

    def validate_repo(self):
        repo_path = self.config.get('repo_path')
        if not repo_path:
            raise RuntimeError(
                f'{self.provenance.name} uses an external official implementation wrapper. '
                f'Set baseline.{self.method_key}.repo_path to a checkout of {self.provenance.repo_url}.')
        repo_path = os.path.abspath(os.path.expanduser(repo_path))
        missing = [path for path in self.provenance.required_files
                   if not os.path.exists(os.path.join(repo_path, path))]
        if missing:
            raise RuntimeError(
                f'{self.provenance.name} repo_path is missing required official files: {missing}')
        expected = self.config.get('expected_commit')
        try:
            actual = subprocess.check_output(
                ['git', '-C', repo_path, 'rev-parse', 'HEAD'], text=True).strip()
        except Exception:
            actual = None
        if expected and actual and not actual.startswith(str(expected)):
            raise RuntimeError(
                f'{self.provenance.name} checkout commit mismatch: expected {expected}, got {actual}')
        self.repo_commit = actual
        return repo_path

    def export_split(self, dataset, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        frames = {
            'train': dataset.train_data,
            'validation': dataset.val_data,
            'test': dataset.test_data,
        }
        for name, frame in frames.items():
            frame[['user_id', 'item_id']].to_csv(
                os.path.join(output_dir, f'{name}.csv'), index=False)
        pd.DataFrame({
            'item_id': list(range(dataset.n_items)),
            'train_frequency': [dataset.item_freq.get(item, 0) for item in range(dataset.n_items)],
        }).to_csv(os.path.join(output_dir, 'item_frequency.csv'), index=False)
        pd.DataFrame({
            'user_id': list(range(dataset.n_users)),
            'train_frequency': [dataset.user_freq.get(user, 0) for user in range(dataset.n_users)],
        }).to_csv(os.path.join(output_dir, 'user_frequency.csv'), index=False)
        warm_items = [item for item in range(dataset.n_items)
                      if dataset.item_freq.get(item, 0) > 0]
        metadata = {
            'method': self.method_key,
            'source_title': self.provenance.source_title,
            'venue': self.provenance.venue,
            'doi': self.provenance.doi,
            'official_repo': self.provenance.repo_url,
            'repo_commit': self.repo_commit,
            'dataset': dataset.name,
            'n_users': dataset.n_users,
            'n_items': dataset.n_items,
            'n_train': int(len(dataset.train_data)),
            'n_validation': int(len(dataset.val_data)),
            'n_test': int(len(dataset.test_data)),
            'split_hash': dataset.split_hash,
            'warm_start_items': warm_items,
        }
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as handle:
            json.dump(metadata, handle, indent=2)
        if hasattr(dataset, 'save_id_mappings'):
            dataset.save_id_mappings(os.path.join(output_dir, 'id_mappings.json'))
        self.prepared_dir = output_dir
        return output_dir

    def _resolve_output_dir(self, dataset):
        root = self.config.get('output_root', 'external_baseline_runs')
        directory = self.config.get('prepared_dir')
        if directory:
            return os.path.abspath(os.path.expanduser(directory))
        return os.path.abspath(os.path.join(root, self.method_key, dataset.name, dataset.split_hash[:12]))

    def _resolve_command(self, repo_path, prepared_dir, output_dir):
        command = self.config.get('command')
        if not command:
            raise RuntimeError(
                f'{self.provenance.name} wrapper prepared data at {prepared_dir}, but no external command is configured. '
                f'Set baseline.{self.method_key}.command to a command that runs the official code and writes metrics.json or scores.npy.')
        if isinstance(command, str):
            command = shlex.split(command)
        return [str(part).replace('{repo_path}', repo_path)
                .replace('{data_dir}', prepared_dir)
                .replace('{output_dir}', output_dir)
                .replace('{python}', sys.executable)
                for part in command]

    def _run_command(self, repo_path, prepared_dir, output_dir):
        command = self._resolve_command(repo_path, prepared_dir, output_dir)
        env = os.environ.copy()
        env.update({
            'DUAL2FAIR_EXTERNAL_DATA_DIR': prepared_dir,
            'DUAL2FAIR_EXTERNAL_OUTPUT_DIR': output_dir,
            'DUAL2FAIR_EXTERNAL_METHOD': self.method_key,
        })
        extra_env = self.config.get('env') or {}
        env.update({str(key): str(value) for key, value in extra_env.items()})
        subprocess.run(command, cwd=repo_path, env=env, check=True)

    def _load_json_metrics(self, path):
        with open(path) as handle:
            metrics = json.load(handle)
        required = {'NDCG', 'Hit', 'DUF', 'DIF'}
        missing = required - set(metrics)
        if missing:
            raise RuntimeError(f'External metrics file is missing keys: {sorted(missing)}')
        metrics.setdefault('UIF', None)
        metrics['external_method'] = self.method_key
        metrics['external_repo'] = self.provenance.repo_url
        metrics['external_repo_commit'] = self.repo_commit
        return metrics

    def _evaluate_scores(self, score_path, evaluator):
        scores = np.load(score_path)
        if scores.ndim != 2:
            raise RuntimeError('External scores.npy must have shape [n_users, n_items]')
        result = evaluator.evaluate(model=ExternalScoreMatrixModel(scores))
        result['external_method'] = self.method_key
        result['external_repo'] = self.provenance.repo_url
        result['external_repo_commit'] = self.repo_commit
        return result

    def collect_results(self, output_dir, evaluator, split):
        metrics_path = self.config.get(f'{split}_metrics_path') or self.config.get('metrics_path')
        score_path = self.config.get(f'{split}_score_matrix_path') or self.config.get('score_matrix_path')
        metrics_path = metrics_path or os.path.join(output_dir, f'{split}_metrics.json')
        score_path = score_path or os.path.join(output_dir, f'{split}_scores.npy')
        if os.path.exists(metrics_path):
            return self._load_json_metrics(metrics_path)
        if os.path.exists(score_path):
            return self._evaluate_scores(score_path, evaluator)
        raise RuntimeError(
            f'{self.provenance.name} external run did not produce {metrics_path} or {score_path}. '
            'Configure the official adapter command to export split metrics or full score matrices.')

    def run(self, dataset, val_evaluator, test_evaluator):
        repo_path = self.validate_repo()
        output_dir = self._resolve_output_dir(dataset)
        prepared_dir = self.export_split(dataset, os.path.join(output_dir, 'prepared_data'))
        if self.config.get('skip_command', False):
            raise RuntimeError(
                f'{self.provenance.name} wrapper prepared data at {prepared_dir}; skip_command=True prevents claiming results.')
        self._run_command(repo_path, prepared_dir, output_dir)
        try:
            self._best_validation_results = self.collect_results(output_dir, val_evaluator, 'validation')
        except RuntimeError:
            self._best_validation_results = None
        return self.collect_results(output_dir, test_evaluator, 'test')


class ESAM(OfficialExternalBaseline):
    method_key = 'esam'


class MGL(OfficialExternalBaseline):
    method_key = 'mgl'
