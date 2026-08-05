import os
import sys
import json
import yaml
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run import (set_seed, load_config, train_standard, train_dual2fair,
                 train_baseline, load_dataset, get_device)
from baseline import BASELINES, CATEGORIES, PROCESSING_TYPES


def run_all_experiments(config_path='config/default.yaml', datasets=None,
                       backbones=None, methods=None, eval_mode='sampled',
                       gpu=0, seed=42):
    config = load_config(config_path)
    device = get_device(gpu)

    if datasets is None:
        datasets = ['epinions', 'movielens', 'gowalla']
    if backbones is None:
        backbones = ['neumf', 'vaecf', 'lightgcn']
    if methods is None:
        methods = ['standard', 'dual2fair', 'ufr', 'hyperuof', 'dpr', 'fairdual',
                   'cpfair', 'multifr', 'ada2fair', 'fair', 'fairsort']

    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        set_seed(seed)
        ds_config = config.get('dataset', {}).get(dataset_name, {})
        dataset = load_dataset(dataset_name,
                               min_ui=ds_config.get('min_user_interactions', 5),
                               min_ii=ds_config.get('min_item_interactions', 5))
        print(f"Users: {dataset.n_users}, Items: {dataset.n_items}, "
              f"Interactions: {len(dataset.interactions)}")

        for backbone_name in backbones:
            print(f"\n--- Standard {backbone_name} ---")
            save_path = os.path.join(save_dir, f"{dataset_name}_{backbone_name}_standard.pt")
            backbone, user_embs, item_embs, ndcg, results = train_standard(
                dataset, backbone_name, config, device, eval_mode=eval_mode,
                save_path=save_path)

            key = f"{dataset_name}_{backbone_name}_standard"
            validation_results = getattr(backbone, '_best_validation_results', results)
            baseline_cfg = dict(config.get('evaluation', {}))
            baseline_cfg['baseline_duf'] = validation_results['DUF']
            baseline_cfg['baseline_dif'] = validation_results['DIF']
            w1 = baseline_cfg.get('uif_w1', 0.5)
            w2 = baseline_cfg.get('uif_w2', 0.5)
            if results['NDCG'] > 0 and validation_results['DUF'] > 0 and validation_results['DIF'] > 0:
                results['UIF'] = (w1 * results['DUF'] / validation_results['DUF']
                                  + w2 * results['DIF'] / validation_results['DIF']) / results['NDCG']
            all_results[key] = {
                'NDCG': round(results['NDCG'], 4),
                'Hit': round(results['Hit'], 4),
                'DUF': round(results['DUF'], 4),
                'DIF': round(results['DIF'], 4),
                'UIF': round(results['UIF'], 4),
                'UIF_baseline_DUF': validation_results['DUF'],
                'UIF_baseline_DIF': validation_results['DIF'],
            }
            config.setdefault('evaluation_by_backbone', {})[
                f"{dataset_name}_{backbone_name}"] = baseline_cfg

        for backbone_name in backbones:
            for method_name in methods:
                if method_name == 'standard':
                    continue

                print(f"\n--- {method_name} on {backbone_name} ---")
                save_path = os.path.join(save_dir,
                                         f"{dataset_name}_{backbone_name}_{method_name}.pt")
                run_config = dict(config)
                eval_config = dict(config.get('evaluation', {}))
                eval_config.update(config.get('evaluation_by_backbone', {}).get(
                    f"{dataset_name}_{backbone_name}", {}))
                run_config['evaluation'] = eval_config

                try:
                    if method_name == 'dual2fair':
                        lambda_range = config.get('grid_search', {}).get(
                            'lambda1_range', [0.01, 0.05, 0.1, 0.5, 1, 10])
                        lambda2_range = config.get('grid_search', {}).get(
                            'lambda2_range', lambda_range)
                        best_ndcg = 0.0
                        best_lambda1 = 0.1
                        best_lambda2 = 0.1

                        for l1 in lambda_range:
                            for l2 in lambda2_range:
                                print(f"  λ1={l1}, λ2={l2}")
                                backbone, user_embs, item_embs, ndcg, results = train_dual2fair(
                                    dataset, backbone_name, run_config, device, l1, l2,
                                    eval_mode=eval_mode)
                                if ndcg > best_ndcg:
                                    best_ndcg = ndcg
                                    best_lambda1 = l1
                                    best_lambda2 = l2

                        backbone, user_embs, item_embs, ndcg, results = train_dual2fair(
                            dataset, backbone_name, run_config, device,
                            best_lambda1, best_lambda2, eval_mode=eval_mode,
                            save_path=save_path)

                        key = f"{dataset_name}_{backbone_name}_dual2fair"
                        all_results[key] = {
                            'NDCG': round(results['NDCG'], 4),
                            'Hit': round(results['Hit'], 4),
                            'DUF': round(results['DUF'], 4),
                            'DIF': round(results['DIF'], 4),
                            'UIF': round(results['UIF'], 4),
                            'lambda1': best_lambda1,
                            'lambda2': best_lambda2,
                        }
                    else:
                        backbone, results = train_baseline(
                            dataset, backbone_name, method_name, run_config, device,
                            eval_mode=eval_mode, save_path=save_path)
                        key = f"{dataset_name}_{backbone_name}_{method_name}"
                        all_results[key] = {
                            'NDCG': round(results.get('NDCG', 0), 4),
                            'Hit': round(results.get('Hit', 0), 4),
                            'DUF': round(results.get('DUF', 0), 4),
                            'DIF': round(results.get('DIF', 0), 4),
                            'UIF': round(results.get('UIF', 0), 4),
                        }
                except Exception as e:
                    traceback.print_exc()
                    print(f"Error: {e}")
                    key = f"{dataset_name}_{backbone_name}_{method_name}"
                    all_results[key] = {'error': str(e)}

    results_path = os.path.join(results_dir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")

    print(f"\n{'='*80}")
    print("RESULTS TABLE")
    print(f"{'='*80}")
    for dataset_name in datasets:
        print(f"\n--- {dataset_name} ---")
        for backbone_name in backbones:
            print(f"\n  {backbone_name.upper()}:")
            print(f"  {'Method':<15} {'NDCG':<10} {'DUF':<10} {'DIF':<10} {'UIF':<10}")
            for method_name in methods:
                key = f"{dataset_name}_{backbone_name}_{method_name}"
                if key in all_results and 'error' not in all_results[key]:
                    r = all_results[key]
                    print(f"  {method_name:<15} {r['NDCG']:<10} {r['DUF']:<10} {r['DIF']:<10} {r['UIF']:<10}")

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run all Dual2Fair experiments')
    parser.add_argument('--config', type=str, default='config/default.yaml')
    parser.add_argument('--datasets', nargs='+', default=['movielens'])
    parser.add_argument('--backbones', nargs='+', default=['lightgcn'])
    parser.add_argument('--methods', nargs='+',
                        default=['standard', 'dual2fair', 'ufr', 'hyperuof', 'dpr',
                                 'fairdual', 'cpfair', 'multifr', 'ada2fair', 'fair', 'fairsort'])
    parser.add_argument('--eval_mode', type=str, default='sampled',
                        choices=['sampled', 'full'])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    run_all_experiments(args.config, args.datasets, args.backbones, args.methods,
                       eval_mode=args.eval_mode, gpu=args.gpu, seed=args.seed)
