import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
import time
import json
import copy
from collections import defaultdict

os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset_utils import Dataset, load_dataset
from models.backbone import get_backbone, NeuMF, VAECF, LightGCN
from models.dual2fair import Dual2Fair
from models.dual2fair.hierarchical_opt import HierarchicalAlternatingOptimizer
from evaluation.evaluator import Evaluator
from baselines import BASELINES, CATEGORIES, PROCESSING_TYPES


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs', 'default.yaml')


def _deep_update(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path=None):
    path = config_path or DEFAULT_CONFIG_PATH
    if path == 'configs/default.yaml':
        path = DEFAULT_CONFIG_PATH
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def apply_config_override(config, override_path):
    if not override_path:
        return config
    with open(override_path, 'r') as f:
        override = yaml.safe_load(f)
    return _deep_update(config, override)


def apply_uif_reference_file(config, path, dataset_name=None, backbone_name=None):
    if not path:
        return config
    with open(path, 'r') as handle:
        reference = yaml.safe_load(handle)
    if dataset_name and reference.get('dataset') not in {None, dataset_name}:
        raise ValueError('UIF reference dataset mismatch')
    if backbone_name and reference.get('backbone') not in {None, backbone_name}:
        raise ValueError('UIF reference backbone mismatch')
    refs = config.setdefault('evaluation', {}).setdefault('uif_references', {})
    for key in ('val', 'test'):
        if key not in reference:
            raise ValueError(f'UIF reference file missing {key} block')
        refs[key] = {'DUF': reference[key]['DUF'], 'DIF': reference[key]['DIF']}
    return config


def sample_negatives(dataset, n_neg=1):
    interactions = dataset.get_train_interactions()
    user_ids, pos_items, neg_items = [], [], []
    for u, i in interactions:
        user_ids.append(u)
        pos_items.append(i)
        pos_set = dataset.user_items.get(u, set())
        for _ in range(n_neg):
            neg = np.random.randint(0, dataset.n_items)
            while neg in pos_set:
                neg = np.random.randint(0, dataset.n_items)
            neg_items.append(neg)
    return torch.LongTensor(user_ids), torch.LongTensor(pos_items), torch.LongTensor(neg_items)


def sample_bce_negatives(dataset, n_neg=99):
    all_items = np.arange(dataset.n_items)
    user_ids, item_ids, labels = [], [], []
    for u in range(dataset.n_users):
        pos_set = dataset.user_items.get(u, set())
        if len(pos_set) == 0:
            continue
        for pos_i in pos_set:
            user_ids.append(u)
            item_ids.append(pos_i)
            labels.append(1.0)
        candidate_pool = np.setdiff1d(all_items, list(pos_set))
        if len(candidate_pool) < n_neg:
            sampled_neg = candidate_pool
        else:
            sampled_neg = np.random.choice(candidate_pool, size=n_neg, replace=False)
        for neg_i in sampled_neg:
            user_ids.append(u)
            item_ids.append(neg_i)
            labels.append(0.0)
    return torch.LongTensor(user_ids), torch.LongTensor(item_ids), torch.FloatTensor(labels)


def get_device(gpu_id=0):
    if gpu_id < 0:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _make_evaluator(dataset, config, device, split='test'):
    eval_config = config.get('evaluation', {})
    references = eval_config.get('uif_references', {}).get(split, {})
    return Evaluator(
        dataset, k=eval_config.get('top_k', 10), device=device, split=split,
        user_batch_size=eval_config.get('user_batch_size', 64),
        baseline_duf=references.get('DUF'), baseline_dif=references.get('DIF'),
        uif_w1=eval_config.get('uif_w1', 0.5),
        uif_w2=eval_config.get('uif_w2', 0.5),
        eps0=config.get('dual2fair', {}).get('eps0', 1e-8),
        require_uif_reference=eval_config.get('require_uif_reference', True))


def init_backbone(backbone_name, dataset, config, device):
    model_config = config.get('model', {})
    backbone_config = config.get('backbone', {}).get(backbone_name, {})
    emb_dim = model_config.get('embedding_dim', 64)

    valid_keys_map = {
        'neumf': {'mlp_layers'},
        'vaecf': {'encoder_hidden_dims', 'decoder_hidden_dims', 'dropout', 'anneal_cap', 'total_anneal_steps'},
        'lightgcn': {'n_layers'},
    }
    valid_keys = valid_keys_map.get(backbone_name, set())
    filtered_config = {k: v for k, v in backbone_config.items() if k in valid_keys}

    backbone = get_backbone(backbone_name, dataset.n_users, dataset.n_items,
                            embedding_dim=emb_dim, **filtered_config)
    backbone.to(device)

    if backbone_name == 'lightgcn':
        backbone.set_adj_mat(dataset.adj_mat)
    elif backbone_name == 'vaecf':
        backbone.set_interact_mat(dataset.interact_mat)

    return backbone


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return float(max(0.0, num_training_steps - current_step)) / float(max(1, num_training_steps - num_warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def clip_gradients(model, max_norm=1.0):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def manual_param_step(parameters, scale):
    with torch.no_grad():
        for param in parameters:
            if param.grad is not None:
                param.add_(param.grad, alpha=scale)


def train_epoch_bpr(backbone, dataset, optimizer, device, batch_size=4096, n_neg=1, scheduler=None, clip_norm=1.0):
    backbone.train()
    if hasattr(backbone, '_clear_cache'):
        backbone._clear_cache()

    user_ids, pos_items, neg_items = sample_negatives(dataset, n_neg)
    n_samples = len(user_ids)
    perm = torch.randperm(n_samples)
    user_ids = user_ids[perm]
    pos_items = pos_items[perm]
    neg_items = neg_items[perm]
    total_loss = 0.0
    n_batches = 0

    for bs in range(0, n_samples, batch_size):
        be = min(bs + batch_size, n_samples)
        u_b = user_ids[bs:be].to(device)
        p_b = pos_items[bs:be].to(device)
        n_b = neg_items[bs:be].to(device)
        optimizer.zero_grad()
        loss = backbone.bpr_loss(u_b, p_b, n_b)
        loss.backward()
        if clip_norm > 0:
            clip_gradients(backbone, clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def train_epoch_bce(backbone, dataset, optimizer, device, batch_size=65536, n_neg=99, scheduler=None, clip_norm=1.0):
    backbone.train()
    if hasattr(backbone, '_clear_cache'):
        backbone._clear_cache()

    user_ids, item_ids, labels = sample_bce_negatives(dataset, n_neg)
    n_samples = len(user_ids)
    perm = torch.randperm(n_samples)
    user_ids = user_ids[perm]
    item_ids = item_ids[perm]
    labels = labels[perm]
    total_loss = 0.0
    n_batches = 0

    for bs in range(0, n_samples, batch_size):
        be = min(bs + batch_size, n_samples)
        u_b = user_ids[bs:be].to(device)
        i_b = item_ids[bs:be].to(device)
        l_b = labels[bs:be].to(device)
        optimizer.zero_grad()
        loss = backbone.bce_loss(u_b, i_b, l_b)
        loss.backward()
        if clip_norm > 0:
            clip_gradients(backbone, clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def train_epoch_vaecf(backbone, dataset, optimizer, device, batch_size=512, clip_norm=1.0):
    backbone.train()
    if hasattr(backbone, '_clear_cache'):
        backbone._clear_cache()
    user_perm = np.random.permutation(dataset.n_users)
    total_loss = 0.0
    n_batches = 0
    for bs in range(0, dataset.n_users, batch_size):
        be = min(bs + batch_size, dataset.n_users)
        batch_users = user_perm[bs:be]
        user_rows = dataset.interact_mat[batch_users].toarray()
        user_rows = torch.from_numpy(user_rows).float()
        optimizer.zero_grad()
        loss = backbone.train_batch(user_rows, device)
        loss.backward()
        if clip_norm > 0:
            clip_gradients(backbone, clip_norm)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(1, n_batches)


def evaluate_model(backbone, evaluator, backbone_name, eval_mode='full'):
    backbone.eval()
    if backbone_name == 'lightgcn':
        backbone._update_cache()
    if eval_mode == 'sampled':
        return evaluator.sampled_evaluate(model=backbone)
    else:
        return evaluator.evaluate(model=backbone)


def evaluate_embeddings_or_model(evaluator, eval_mode='full', model=None,
                                 user_embeddings=None, item_embeddings=None):
    if eval_mode == 'sampled':
        return evaluator.sampled_evaluate(model=model,
                                          user_embeddings=user_embeddings,
                                          item_embeddings=item_embeddings)
    return evaluator.evaluate(model=model,
                              user_embeddings=user_embeddings,
                              item_embeddings=item_embeddings)


def compute_baseline_fair_loss(baseline_name, baseline_obj, backbone, dataset,
                               adv_users, disadv_users, user_ids=None,
                               pos_items=None, neg_items=None):
    user_embs = backbone.get_user_embeddings()
    item_embs = backbone.get_item_embeddings()
    if baseline_name == 'dpr':
        return baseline_obj.compute_dpr_loss(user_embs, item_embs,
                                             dataset.item_freq, dataset.n_items)
    if baseline_name == 'multifr':
        item_fair = (baseline_obj.compute_exposure_fairness_loss(user_embs, item_embs,
                                                                 dataset.item_freq, dataset.n_items)
                     if hasattr(baseline_obj, 'compute_exposure_fairness_loss')
                     else baseline_obj.compute_item_fairness_loss(item_embs, dataset.item_freq, dataset.n_items))
        return baseline_obj.compute_user_fairness_loss(user_embs, adv_users, disadv_users) + item_fair
    if baseline_name == 'fairdual':
        if hasattr(baseline_obj, 'compute_exposure_fairness_loss'):
            return baseline_obj.compute_exposure_fairness_loss(
                user_embs, item_embs, dataset.item_freq, dataset.n_items)
        return baseline_obj.compute_item_fairness_loss(item_embs, dataset.item_freq, dataset.n_items)
    if baseline_name == 'ada2fair':
        user_fair = baseline_obj.compute_user_fairness_loss(user_embs, adv_users, disadv_users)
        item_fair = (baseline_obj.compute_exposure_fairness_loss(user_embs, item_embs,
                                                                 dataset.item_freq, dataset.n_items)
                     if hasattr(baseline_obj, 'compute_exposure_fairness_loss')
                     else baseline_obj.compute_item_fairness_loss(item_embs, dataset.item_freq, dataset.n_items))
        return user_fair + item_fair
    if baseline_name == 'fair':
        return baseline_obj.compute_online_fairness_loss(
            user_embs, item_embs, adv_users, disadv_users, dataset.item_freq, dataset.n_items)
    return torch.tensor(0.0, device=user_embs.device)


def train_standard(dataset, backbone_name, config, device, eval_mode='full',
                   loss_type='bce', save_path=None, evaluation_stage='both'):
    backbone = init_backbone(backbone_name, dataset, config, device)
    backbone._backbone_name_hint = backbone_name

    model_config = config.get('model', {})
    lr = model_config.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr,
                                 weight_decay=model_config.get('weight_decay', 1e-5))

    val_evaluator = _make_evaluator(dataset, config, device, split='val')
    test_evaluator = _make_evaluator(dataset, config, device, split='test')

    best_ndcg = 0.0
    best_results = None
    best_state = None
    patience = 0
    max_epochs = model_config.get('max_epochs', 200)
    early_stop = model_config.get('early_stop_patience', 50)
    batch_size = model_config.get('batch_size', 65536)
    n_neg = model_config.get('n_neg_bce', 4) if loss_type == 'bce' else model_config.get('n_neg', 1)

    warmup_ratio = model_config.get('warmup_ratio', 0.1)
    n_train_interactions = len(dataset.get_train_interactions())
    n_steps_per_epoch = max(1, n_train_interactions // batch_size)
    total_steps = max_epochs * n_steps_per_epoch
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(max_epochs):
        t0 = time.time()

        if backbone_name == 'vaecf':
            avg_loss = train_epoch_vaecf(backbone, dataset, optimizer, device, batch_size,
                                         model_config.get('gradient_clip_val', 1.0))
        elif loss_type == 'bce':
            avg_loss = train_epoch_bce(backbone, dataset, optimizer, device, batch_size, n_neg, scheduler)
        else:
            avg_loss = train_epoch_bpr(backbone, dataset, optimizer, device, batch_size, n_neg, scheduler)

        results = evaluate_model(backbone, val_evaluator, backbone_name, eval_mode)
        t1 = time.time()
        tag = 'S-NDCG' if eval_mode == 'sampled' else 'F-NDCG'
        print(f"[Standard/{backbone_name}] Epoch {epoch+1}: loss={avg_loss:.4f}, "
              f"{tag}={results['NDCG']:.4f}, Hit={results['Hit']:.4f}, "
              f"DUF={results['DUF']:.6f}, DIF={results['DIF']:.6f}, "
              f"time={t1-t0:.1f}s")

        if results['NDCG'] > best_ndcg:
            best_ndcg = results['NDCG']
            best_results = dict(results)
            best_state = copy.deepcopy(backbone.state_dict())
            patience = 0
            if save_path:
                torch.save(backbone.state_dict(), save_path)
        else:
            patience += 1

        if patience >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        backbone.load_state_dict(best_state)
        if hasattr(backbone, '_clear_cache'):
            backbone._clear_cache()
    print(f"[Standard/{backbone_name}] Best val: {best_results}")
    if evaluation_stage == 'validation':
        test_results = dict(best_results)
        test_results['evaluation_stage'] = 'validation'
    else:
        test_results = evaluate_model(backbone, test_evaluator, backbone_name, eval_mode)
        test_results['evaluation_stage'] = 'test'
        print(f"[Standard/{backbone_name}] Test: {test_results}")
    user_embs = backbone.get_user_embeddings().detach().cpu().numpy()
    item_embs = backbone.get_item_embeddings().detach().cpu().numpy()
    backbone._best_validation_results = best_results
    return backbone, user_embs, item_embs, best_ndcg, test_results


def train_dual2fair(dataset, backbone_name, config, device, lambda1=0.1, lambda2=0.1,
                    eval_mode='full', loss_type='bpr', save_path=None,
                    evaluation_stage='both'):
    backbone = init_backbone(backbone_name, dataset, config, device)
    backbone._backbone_name_hint = backbone_name
    model = Dual2Fair(backbone, dataset, config, device, backbone_name).to(device)
    settings = config['dual2fair']
    model_config = config['model']
    controller = HierarchicalAlternatingOptimizer(
        model, accuracy_learning_rate=model_config.get('learning_rate', 1e-3),
        fairness_learning_rate=settings['fairness_learning_rate'],
        mirror_interval=settings['mirror_interval'],
        mirror_alpha1=settings['mirror_alpha1'],
        mirror_alpha2=settings['mirror_alpha2'],
        weight_decay=model_config.get('weight_decay', 0.0))
    val_evaluator = _make_evaluator(dataset, config, device, 'val')
    test_evaluator = _make_evaluator(dataset, config, device, 'test')
    max_epochs = model_config.get('max_epochs', 200)
    patience_limit = model_config.get('early_stop_patience', 50)
    batch_size = model_config.get('batch_size', 4096)
    refresh_epochs = settings['calibration_refresh_epochs']
    strategy = settings.get('optimization_strategy', 'hierarchical_alternating')
    print('Using accuracy-first hierarchical alternating optimization')
    validation_trajectory = []
    best_ndcg, best_results, best_checkpoint = float('-inf'), None, None
    patience = iteration = 0
    model.refresh_calibration_state()
    for epoch in range(max_epochs):
        model.train()
        if backbone_name == 'vaecf':
            users = torch.randperm(dataset.n_users)
            batches = [(users[start:start + batch_size].to(device), None, None)
                       for start in range(0, len(users), batch_size)]
        elif loss_type == 'bce':
            users, items, labels = sample_bce_negatives(dataset, model_config.get('n_neg_bce', 4))
            batches = [(users[start:start + batch_size].to(device),
                        items[start:start + batch_size].to(device),
                        labels[start:start + batch_size].to(device))
                       for start in range(0, len(users), batch_size)]
        else:
            users, positive, negative = sample_negatives(dataset, model_config.get('n_neg', 1))
            batches = [(users[start:start + batch_size].to(device),
                        positive[start:start + batch_size].to(device),
                        negative[start:start + batch_size].to(device))
                       for start in range(0, len(users), batch_size)]
        for first, second, third in batches:
            def accuracy_loss():
                model.build_calibrated_embeddings()
                if backbone_name == 'vaecf':
                    rows = torch.stack([backbone._get_interact_batch(int(user), int(user)+1)[0]
                                        for user in first.cpu().tolist()]).to(device)
                    mean, logvar = backbone._encode(rows)
                    logits = model.scoring_state['calibrated_user_repr'][first] @ model.scoring_state['calibrated_item_repr'].T + backbone.get_item_bias()
                    rec_loss = -torch.sum(rows * torch.log_softmax(logits, 1)) / len(rows) + backbone.kl_loss(mean, logvar) / len(rows)
                else:
                    rec_loss = (model.bce_loss(first, second, third) if loss_type == 'bce'
                                else model.bpr_loss(first, second, third))
                if strategy == 'joint_weighted_sum':
                    rec_loss = rec_loss + lambda1 * model.compute_user_fixed_coupling_loss() + lambda2 * model.compute_item_fixed_coupling_loss()
                return rec_loss
            controller.step_iteration(accuracy_loss)
            iteration += 1
        if (epoch + 1) % refresh_epochs == 0:
            model.refresh_calibration_state()
            if strategy != 'joint_weighted_sum':
                controller.refresh_correction(lambda1, lambda2, strategy)
        model.eval()
        model.build_calibrated_embeddings()
        results = val_evaluator.evaluate(model=model)
        selected = results['NDCG'] > best_ndcg
        if selected:
            for entry in validation_trajectory:
                entry['selected_checkpoint'] = False
        trajectory_entry = {
            'epoch': epoch + 1,
            'validation_NDCG': results['NDCG'],
            'validation_DUF': results['DUF'],
            'validation_DIF': results['DIF'],
            'validation_UIF': results.get('UIF'),
            'strategy': strategy,
            'seed': config.get('seeds', {}).get('model'),
            'selected_checkpoint': selected}
        validation_trajectory.append(trajectory_entry)
        print(f"[Dual2Fair/{backbone_name}] Epoch {epoch+1}: "
              f"NDCG={results['NDCG']:.4f}, DUF={results['DUF']:.6f}, "
              f"DIF={results['DIF']:.6f}, UIF={results.get('UIF')}")
        if selected:
            best_ndcg, best_results, patience = results['NDCG'], dict(results), 0
            best_checkpoint = copy.deepcopy(model.checkpoint_state(
                controller.accuracy_optimizer, controller.fairness_optimizer,
                epoch + 1, iteration, {'validation': best_results}))
            if save_path:
                torch.save(best_checkpoint, save_path)
        else:
            patience += 1
        if patience >= patience_limit:
            break
    if best_checkpoint is None:
        raise RuntimeError('No Dual2Fair checkpoint selected')
    model.load_checkpoint_state(best_checkpoint)
    model.eval(); model.build_calibrated_embeddings()
    if evaluation_stage == 'validation':
        test_results = dict(best_results)
        test_results['evaluation_stage'] = 'validation'
    else:
        test_results = test_evaluator.evaluate(model=model)
        test_results['evaluation_stage'] = 'test'
    output = model.build_calibrated_embeddings()
    model._best_validation_results = best_results
    model._validation_trajectory = validation_trajectory
    return model, output.calibrated_user_representations.detach().cpu().numpy(), output.calibrated_item_representations.detach().cpu().numpy(), best_ndcg, test_results

def train_inprocessing_baseline(dataset, backbone_name, baseline_name, config, device,
                                eval_mode='full', loss_type='bce', save_path=None):
    backbone = init_backbone(backbone_name, dataset, config, device)
    backbone._backbone_name_hint = backbone_name

    model_config = config.get('model', {})
    eval_config = config.get('evaluation', {})
    baseline_config = config.get('baseline', {}).get(baseline_name, {})
    lr = model_config.get('learning_rate', 0.001)
    embedding_dim = model_config.get('embedding_dim', 64)

    adv_users, disadv_users = dataset.get_user_activity_groups(
        config.get('dual2fair', {}).get('sparse_user_ratio', 0.95))
    hot_items, cold_items = dataset.get_hot_cold_items()

    baseline_cls = BASELINES[baseline_name]

    if baseline_name == 'hyperuof':
        baseline_obj = baseline_cls(dataset.n_users, dataset.n_items,
                                    embedding_dim=embedding_dim, device=device)
        all_params = list(backbone.parameters()) + list(baseline_obj.user_emb.parameters()) + list(baseline_obj.item_emb.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=model_config.get('weight_decay', 1e-5))
    elif baseline_name == 'ada2fair':
        baseline_obj = baseline_cls(lambda_user=baseline_config.get('lambda_user', 0.1),
                                    lambda_item=baseline_config.get('lambda_item', 0.1),
                                    embedding_dim=embedding_dim, device=device)
        all_params = list(backbone.parameters()) + list(baseline_obj.weight_generator.parameters())
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=model_config.get('weight_decay', 1e-5))
    elif baseline_name == 'dpr':
        baseline_obj = baseline_cls(lambda_dpr=baseline_config.get('lambda_dpr', 0.1),
                                    reg_s=baseline_config.get('reg_s', 1e-4),
                                    alpha_adv=baseline_config.get('alpha_adv', 0.01),
                                    device=device)
        optimizer = torch.optim.Adam(backbone.parameters(), lr=lr, weight_decay=model_config.get('weight_decay', 1e-5))
    elif baseline_name == 'fairdual':
        baseline_obj = baseline_cls(n_groups=2,
                                    lambda_dual=baseline_config.get('lambda_dual', 0.1),
                                    device=device)
        all_params = list(backbone.parameters()) + [baseline_obj.shadow_prices]
        optimizer = torch.optim.Adam(all_params, lr=lr, weight_decay=model_config.get('weight_decay', 1e-5))
    elif baseline_name == 'multifr':
        baseline_obj = baseline_cls(lambda_user=baseline_config.get('lambda_user', 0.1),
                                    lambda_item=baseline_config.get('lambda_item', 0.1),
                                    device=device)
        optimizer = torch.optim.Adam(backbone.parameters(), lr=lr, weight_decay=model_config.get('weight_decay', 1e-5))
    elif baseline_name == 'fair':
        baseline_obj = baseline_cls(lambda_user=baseline_config.get('lambda_user', 0.1),
                                    lambda_item=baseline_config.get('lambda_item', 0.1),
                                    eta=baseline_config.get('eta', 0.01),
                                    device=device)
        optimizer = torch.optim.Adam(backbone.parameters(), lr=lr, weight_decay=model_config.get('weight_decay', 1e-5))
    elif baseline_name == 'popularity_ips':
        baseline_obj = baseline_cls(
            alpha=baseline_config.get('alpha', 0.5),
            max_weight=baseline_config.get('max_weight', 10.0), device=device)
        optimizer = torch.optim.Adam(backbone.parameters(), lr=lr,
                                     weight_decay=model_config.get('weight_decay', 1e-5))
    else:
        baseline_obj = baseline_cls()
        optimizer = torch.optim.Adam(backbone.parameters(), lr=lr,
                                     weight_decay=model_config.get('weight_decay', 1e-5))

    val_evaluator = _make_evaluator(dataset, config, device, split='val')
    test_evaluator = _make_evaluator(dataset, config, device, split='test')

    best_ndcg = 0.0
    best_results = None
    best_state = None
    patience = 0
    max_epochs = model_config.get('max_epochs', 200)
    early_stop = model_config.get('early_stop_patience', 50)
    batch_size = model_config.get('batch_size', 4096)
    n_neg = model_config.get('n_neg', 1)

    for epoch in range(max_epochs):
        backbone.train()
        t0 = time.time()
        if baseline_name == 'ada2fair' and hasattr(baseline_obj, 'update_fairness_weights'):
            baseline_obj.update_fairness_weights(
                backbone.get_user_embeddings(), backbone.get_item_embeddings(),
                dataset.user_items, dataset.item_freq,
                top_k=baseline_config.get('top_k', 100),
                delta=baseline_config.get('delta', 1e-6),
                provider_eta=baseline_config.get('provider_eta', 1.0),
                user_eta=baseline_config.get('user_eta', 1.0),
                user_batch_size=baseline_config.get('user_batch_size', 512))

        if backbone_name == 'vaecf':
            rec_loss_val = train_epoch_vaecf(backbone, dataset, optimizer, device, batch_size,
                                             model_config.get('gradient_clip_val', 1.0))
            optimizer.zero_grad()
            if baseline_name == 'hyperuof':
                fair_loss = baseline_obj.fair_loss(adv_users, disadv_users)
                total = 0.1 * fair_loss
            else:
                fair_loss = compute_baseline_fair_loss(
                    baseline_name, baseline_obj, backbone, dataset, adv_users, disadv_users)
                total = fair_loss
            total.backward()
            if model_config.get('gradient_clip_val', 1.0) > 0:
                clip_gradients(backbone, model_config.get('gradient_clip_val', 1.0))
            optimizer.step()
            rec_loss_val += float(total.detach().cpu())
        else:
            user_ids, pos_items, neg_items = sample_negatives(dataset, n_neg)
            n_samples = len(user_ids)
            perm = torch.randperm(n_samples)
            user_ids = user_ids[perm]
            pos_items = pos_items[perm]
            neg_items = neg_items[perm]
            total_loss = 0.0
            n_batches = 0

            for bs in range(0, n_samples, batch_size):
                be = min(bs + batch_size, n_samples)
                u_b = user_ids[bs:be].to(device)
                p_b = pos_items[bs:be].to(device)
                n_b = neg_items[bs:be].to(device)

                optimizer.zero_grad()

                if baseline_name == 'hyperuof':
                    h_bpr = baseline_obj.bpr_loss(u_b, p_b, n_b)
                    h_fair = baseline_obj.fair_loss(adv_users, disadv_users)
                    total = h_bpr + 0.1 * h_fair
                elif baseline_name == 'ada2fair':
                    weights = baseline_obj.compute_adaptive_weights(
                        backbone.get_user_embeddings(), backbone.get_item_embeddings(),
                        u_b, p_b)
                    weighted_bpr = baseline_obj.compute_weighted_bpr_loss(
                        backbone, u_b, p_b, n_b, weights)
                    fair_loss = compute_baseline_fair_loss(
                        baseline_name, baseline_obj, backbone, dataset, adv_users, disadv_users)
                    total = weighted_bpr + fair_loss
                elif baseline_name == 'popularity_ips':
                    total = baseline_obj.weighted_bpr_loss(
                        backbone, u_b, p_b, n_b, dataset.item_freq)
                else:
                    rec_loss = backbone.bpr_loss(u_b, p_b, n_b)
                    fair_loss = compute_baseline_fair_loss(
                        baseline_name, baseline_obj, backbone, dataset, adv_users, disadv_users)
                    total = rec_loss + fair_loss

                total.backward()
                optimizer.step()
                total_loss += total.item()
                n_batches += 1

            rec_loss_val = total_loss / max(1, n_batches)

        if baseline_name == 'hyperuof':
            user_eval = baseline_obj.get_user_embeddings().detach().cpu().numpy()
            item_eval = baseline_obj.get_item_embeddings().detach().cpu().numpy()
            results = evaluate_embeddings_or_model(
                val_evaluator, eval_mode, user_embeddings=user_eval, item_embeddings=item_eval)
        else:
            results = evaluate_model(backbone, val_evaluator, backbone_name, eval_mode)
        t1 = time.time()
        tag = 'S-NDCG' if eval_mode == 'sampled' else 'F-NDCG'
        print(f"[{baseline_name}/{backbone_name}] Epoch {epoch+1}: loss={rec_loss_val:.4f}, "
              f"{tag}={results['NDCG']:.4f}, Hit={results['Hit']:.4f}, "
              f"DUF={results['DUF']:.6f}, DIF={results['DIF']:.6f}, "
              f"time={t1-t0:.1f}s")

        if results['NDCG'] > best_ndcg:
            best_ndcg = results['NDCG']
            best_results = dict(results)
            if baseline_name == 'hyperuof':
                best_state = {
                    'user_emb': copy.deepcopy(baseline_obj.user_emb.state_dict()),
                    'item_emb': copy.deepcopy(baseline_obj.item_emb.state_dict()),
                }
            else:
                best_state = copy.deepcopy(backbone.state_dict())
            patience = 0
            if save_path:
                if baseline_name == 'hyperuof':
                    torch.save({
                        'user_emb': baseline_obj.user_emb.state_dict(),
                        'item_emb': baseline_obj.item_emb.state_dict(),
                    }, save_path)
                else:
                    torch.save(backbone.state_dict(), save_path)
        else:
            patience += 1

        if patience >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if baseline_name == 'hyperuof':
        if best_state is not None:
            baseline_obj.user_emb.load_state_dict(best_state['user_emb'])
            baseline_obj.item_emb.load_state_dict(best_state['item_emb'])
        user_eval = baseline_obj.get_user_embeddings().detach().cpu().numpy()
        item_eval = baseline_obj.get_item_embeddings().detach().cpu().numpy()
        test_results = evaluate_embeddings_or_model(
            test_evaluator, eval_mode, user_embeddings=user_eval, item_embeddings=item_eval)
    else:
        if best_state is not None:
            backbone.load_state_dict(best_state)
            if hasattr(backbone, '_clear_cache'):
                backbone._clear_cache()
        test_results = evaluate_model(backbone, test_evaluator, backbone_name, eval_mode)
    print(f"[{baseline_name}] Best val: {best_results}")
    print(f"[{baseline_name}] Test: {test_results}")
    return backbone, test_results


def train_postprocessing_baseline(dataset, backbone_name, baseline_name, config, device,
                                  eval_mode='full', loss_type='bce', save_path=None):
    backbone = init_backbone(backbone_name, dataset, config, device)
    backbone._backbone_name_hint = backbone_name

    model_config = config.get('model', {})
    lr = model_config.get('learning_rate', 0.001)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=lr,
                                 weight_decay=model_config.get('weight_decay', 1e-5))

    val_evaluator = _make_evaluator(dataset, config, device, split='val')
    test_evaluator = _make_evaluator(dataset, config, device, split='test')

    best_ndcg = 0.0
    best_backbone_results = None
    best_state = None
    patience = 0
    max_epochs = model_config.get('max_epochs', 200)
    early_stop = model_config.get('early_stop_patience', 50)
    batch_size = model_config.get('batch_size', 4096)
    n_neg = model_config.get('n_neg', 1)

    for epoch in range(max_epochs):
        backbone.train()
        t0 = time.time()

        if backbone_name == 'vaecf':
            avg_loss = train_epoch_vaecf(backbone, dataset, optimizer, device, batch_size,
                                         model_config.get('gradient_clip_val', 1.0))
        else:
            avg_loss = train_epoch_bpr(backbone, dataset, optimizer, device,
                                       batch_size, n_neg)

        results = evaluate_model(backbone, val_evaluator, backbone_name, eval_mode)
        t1 = time.time()
        tag = 'S-NDCG' if eval_mode == 'sampled' else 'F-NDCG'
        print(f"[{baseline_name}-backbone/{backbone_name}] Epoch {epoch+1}: loss={avg_loss:.4f}, "
              f"{tag}={results['NDCG']:.4f}, time={t1-t0:.1f}s")

        if results['NDCG'] > best_ndcg:
            best_ndcg = results['NDCG']
            best_backbone_results = dict(results)
            best_state = copy.deepcopy(backbone.state_dict())
            patience = 0
            if save_path:
                torch.save(backbone.state_dict(), save_path)
        else:
            patience += 1

        if patience >= early_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    backbone.eval()
    if best_state is not None:
        backbone.load_state_dict(best_state)
        if hasattr(backbone, '_clear_cache'):
            backbone._clear_cache()
    if backbone_name == 'lightgcn':
        backbone._update_cache()

    user_embs = backbone.get_user_embeddings().detach().cpu().numpy()
    item_embs = backbone.get_item_embeddings().detach().cpu().numpy()

    baseline_cls = BASELINES[baseline_name]
    eval_config = config.get('evaluation', {})
    baseline_config = config.get('baseline', {}).get(baseline_name, {})
    adv_users, disadv_users = dataset.get_user_activity_groups(
        config.get('dual2fair', {}).get('sparse_user_ratio', 0.95))
    hot_items, cold_items = dataset.get_hot_cold_items()
    k = eval_config.get('top_k', 10)

    if baseline_name == 'ufr':
        user_groups = {}
        for u in adv_users:
            user_groups[u] = 'advantaged'
        for u in disadv_users:
            user_groups[u] = 'disadvantaged'
        baseline_obj = baseline_cls(k=k, delta=baseline_config.get('delta', 0.05))
        reranked = baseline_obj.rerank(
            user_embs, item_embs, test_evaluator.heldout,
            dataset.user_items, user_groups, k=k, device=device)
    elif baseline_name == 'cpfair':
        item_groups = {}
        for i in hot_items:
            item_groups[i] = 'hot'
        for i in cold_items:
            item_groups[i] = 'cold'
        baseline_obj = baseline_cls(alpha=baseline_config.get('alpha', 0.5),
                                    k=k,
                                    utility_weight=baseline_config.get('utility_weight', 0.8))
        reranked = baseline_obj.rerank(
            user_embs, item_embs, test_evaluator.heldout,
            dataset.user_items, item_groups, k=k)
    elif baseline_name == 'fairsort':
        baseline_obj = baseline_cls(k=k,
                                    min_utility=baseline_config.get('min_utility', 0.8),
                                    search_steps=baseline_config.get('search_steps', 6))
        reranked = baseline_obj.rerank(
            user_embs, item_embs, test_evaluator.heldout,
            dataset.user_items, k=k)
    else:
        reranked = {}

    if eval_mode == 'sampled':
        from evaluation.metrics import compute_duf, compute_uif
        sampled_candidates = build_sampled_candidates(
            test_evaluator.heldout, dataset.get_test_neg_items(),
            dataset.user_items, dataset.n_items, eval_config.get('n_neg_sampled', 99))
        ndcg_scores_full, mean_ndcg, mean_hit = compute_sampled_metrics_from_reranked(
            reranked, sampled_candidates, user_embs, item_embs, k)
        dif_val = compute_sampled_dif_from_reranked_embeddings(
            reranked, sampled_candidates, user_embs, item_embs, k)
        duf_val = compute_duf(ndcg_scores_full)
        test_refs = eval_config.get('uif_references', {}).get('test', {})
        uif_val = None
        if test_refs.get('DUF') is not None and test_refs.get('DIF') is not None:
            uif_val = compute_uif(ndcg_scores_full, dif_val,
                              baseline_duf=test_refs.get('DUF'),
                              baseline_dif=test_refs.get('DIF'),
                              w1=eval_config.get('uif_w1', 0.5),
                              w2=eval_config.get('uif_w2', 0.5))
        final_results = {
            'NDCG': mean_ndcg, 'Hit': mean_hit,
            'DUF': duf_val, 'DIF': dif_val,
            'UIF': uif_val
        }
    else:
        from evaluation.metrics import compute_duf, compute_uif
        ndcg_scores, mean_ndcg, mean_hit = compute_ndcg_full_from_reranked(
            reranked, test_evaluator.heldout, k)
        dif_val = compute_dif_from_reranked_embeddings(
            reranked, user_embs, item_embs, test_evaluator.heldout, dataset.user_items, k)
        duf_val = compute_duf(ndcg_scores)
        test_refs = eval_config.get('uif_references', {}).get('test', {})
        uif_val = None
        if test_refs.get('DUF') is not None and test_refs.get('DIF') is not None:
            uif_val = compute_uif(ndcg_scores, dif_val,
                              baseline_duf=test_refs.get('DUF'),
                              baseline_dif=test_refs.get('DIF'),
                              w1=eval_config.get('uif_w1', 0.5),
                              w2=eval_config.get('uif_w2', 0.5))
        final_results = {
            'NDCG': mean_ndcg, 'Hit': mean_hit,
            'DUF': duf_val, 'DIF': dif_val,
            'UIF': uif_val
        }

    print(f"[{baseline_name}] Best val: {best_backbone_results}")
    print(f"[{baseline_name}] Test: {final_results}")
    return backbone, final_results


def compute_ndcg_full_from_reranked(reranked_lists, test_dict, k=10):
    from evaluation.metrics import ndcg_at_k, hit_ratio_at_k
    ndcg_scores, hit_scores = {}, {}
    for uid, test_items in test_dict.items():
        if isinstance(test_items, (int, np.integer)):
            test_items = {int(test_items)}
        elif not isinstance(test_items, set):
            test_items = set(test_items)
        ranked = reranked_lists.get(uid, [])
        ndcg_scores[uid] = ndcg_at_k(ranked, test_items, k)
        hit_scores[uid] = hit_ratio_at_k(ranked, test_items, k)
    mean_ndcg = np.mean(list(ndcg_scores.values())) if ndcg_scores else 0.0
    mean_hit = np.mean(list(hit_scores.values())) if hit_scores else 0.0
    return ndcg_scores, mean_ndcg, mean_hit


def build_sampled_candidates(test_dict, test_neg_items, train_user_items, n_items, n_neg=99):
    sampled_candidates = {}
    all_items = np.arange(n_items)
    for uid, test_items in test_dict.items():
        if isinstance(test_items, (int, np.integer)):
            pos_item = int(test_items)
            test_items_set = {pos_item}
        else:
            test_items_set = set(test_items)
            pos_item = int(next(iter(test_items_set)))

        neg_items = test_neg_items.get(uid) if test_neg_items is not None else None
        if neg_items is None:
            blocked = set(train_user_items.get(uid, set())) | test_items_set
            candidate_pool = np.array([item for item in all_items if item not in blocked], dtype=np.int64)
            if len(candidate_pool) <= n_neg:
                neg_items = candidate_pool.tolist()
            else:
                neg_items = np.random.choice(candidate_pool, size=n_neg, replace=False).tolist()

        sampled_candidates[uid] = [pos_item] + [int(item) for item in neg_items]
    return sampled_candidates


def compute_sampled_metrics_from_reranked(reranked_lists, sampled_candidates,
                                          user_embs, item_embs, k=10):
    from sklearn.metrics import ndcg_score as sklearn_ndcg_score
    ndcg_scores, hit_scores = {}, {}
    item_embs_t = torch.from_numpy(item_embs).float() if isinstance(item_embs, np.ndarray) else item_embs
    user_embs_t = torch.from_numpy(user_embs).float() if isinstance(user_embs, np.ndarray) else user_embs
    for uid, eval_items in sampled_candidates.items():
        if not eval_items:
            continue
        labels = np.zeros(len(eval_items), dtype=np.float32)
        labels[0] = 1.0
        eval_set = set(eval_items)
        selected = [item for item in reranked_lists.get(uid, []) if item in eval_set]
        with torch.no_grad():
            model_scores = (item_embs_t[eval_items] @ user_embs_t[uid]).cpu().numpy()
        score_by_item = {item: float(score) for item, score in zip(eval_items, model_scores)}
        remaining = [item for item in eval_items if item not in selected]
        remaining.sort(key=lambda item: score_by_item[item], reverse=True)
        final_rank = selected + remaining
        rank_pos = {item: pos for pos, item in enumerate(final_rank)}
        scores = np.array([-rank_pos[item] for item in eval_items], dtype=np.float32)
        ndcg_scores[uid] = sklearn_ndcg_score([labels], [scores], k=k)
        top_k_idx = np.argsort(-scores)[:k]
        hit_scores[uid] = 1.0 if 0 in top_k_idx else 0.0
    mean_ndcg = np.mean(list(ndcg_scores.values())) if ndcg_scores else 0.0
    mean_hit = np.mean(list(hit_scores.values())) if hit_scores else 0.0
    return ndcg_scores, mean_ndcg, mean_hit


def compute_rank_dif_from_reranked(reranked_lists, test_dict, n_items, k=10, eps0=1e-8):
    from evaluation.metrics import compute_rank_dif
    exposure = np.zeros(n_items, dtype=np.float64)
    relevance = np.zeros(n_items, dtype=np.float64)
    evaluated = np.zeros(n_items, dtype=bool)
    for uid in test_dict:
        ranked = reranked_lists.get(int(uid), [])
        if not len(ranked):
            continue
        ranked = np.asarray(ranked, dtype=np.int64)
        ranked = ranked[ranked < n_items]
        if not len(ranked):
            continue
        exposure[ranked[:k]] += 1.0
        count = len(ranked)
        relevance[ranked] += (count - np.arange(1, count + 1) + 1) / count
        evaluated[ranked] = True
    return compute_rank_dif(exposure, relevance, evaluated, eps0)


def compute_dif_from_reranked_embeddings(reranked_lists, user_embs, item_embs, test_dict,
                                         train_user_items, k=10, eps0=1e-8):
    n_items = item_embs.shape[0]
    return compute_rank_dif_from_reranked(reranked_lists, test_dict, n_items, k, eps0)


def compute_sampled_dif_from_reranked_embeddings(reranked_lists, sampled_candidates,
                                                 user_embs, item_embs, k=10, eps0=1e-8):
    import warnings
    warnings.warn('sampled DIF is deprecated; use full-catalog rank-based DIF', DeprecationWarning)
    return compute_rank_dif_from_reranked(reranked_lists, sampled_candidates, item_embs.shape[0], k, eps0)


def train_external_baseline(dataset, baseline_name, config, device):
    baseline_config = config.get('baseline', {}).get(baseline_name, {})
    baseline = BASELINES[baseline_name](baseline_config)
    val_evaluator = _make_evaluator(dataset, config, device, split='val')
    test_evaluator = _make_evaluator(dataset, config, device, split='test')
    results = baseline.run(dataset, val_evaluator, test_evaluator)
    return baseline, results


def train_baseline(dataset, backbone_name, baseline_name, config, device,
                   eval_mode='full', loss_type='bce', save_path=None):
    proc_type = PROCESSING_TYPES.get(baseline_name, 'in-processing')
    if proc_type == 'external-wrapper':
        return train_external_baseline(dataset, baseline_name, config, device)
    if proc_type == 'unavailable-external':
        raise NotImplementedError(
            f'{baseline_name} exact implementation is not available locally; '
            'provide an official adapter before running reproduction.')
    if proc_type == 'post-processing':
        return train_postprocessing_baseline(dataset, backbone_name, baseline_name,
                                             config, device, eval_mode, loss_type, save_path)
    else:
        return train_inprocessing_baseline(dataset, backbone_name, baseline_name,
                                           config, device, eval_mode, loss_type, save_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Dual2Fair: Decoupled User-Item Representation Alignment')
    parser.add_argument('--dataset', type=str, default='movielens',
                        choices=['demo', 'movielens', 'epinions', 'gowalla'])
    parser.add_argument('--backbone', type=str, default='lightgcn',
                        choices=['neumf', 'vaecf', 'lightgcn'])
    parser.add_argument('--method', type=str, default='standard',
                        choices=['standard', 'dual2fair', 'ufr', 'hyperuof', 'dpr',
                                 'fairdual', 'cpfair', 'multifr', 'ada2fair', 'fair',
                                 'fairsort', 'popularity_ips', 'esam', 'mgl'])
    parser.add_argument('--lambda1', type=float, default=0.1)
    parser.add_argument('--lambda2', type=float, default=0.1)
    parser.add_argument('--alignment_mode', type=str, default=None,
                        choices=['ot', 'hard', 'mmd'])
    parser.add_argument('--eval_mode', type=str, default='full',
                        choices=['sampled', 'full'])
    parser.add_argument('--evaluation-stage', type=str, default='both',
                        choices=['validation', 'test', 'both'])
    parser.add_argument('--loss_type', type=str, default='bpr',
                        choices=['bce', 'bpr'])
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--uif-reference-file', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--seeds', type=int, nargs='+', default=None)
    parser.add_argument('--split-seed', type=int, default=2026)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output-suffix', type=str, default='')
    parser.add_argument('--allow-missing-uif-reference', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config)
    apply_uif_reference_file(config, args.uif_reference_file, args.dataset, args.backbone)
    if args.alignment_mode is not None:
        config.setdefault('dual2fair', {})['alignment_mode'] = args.alignment_mode
    if args.allow_missing_uif_reference or (args.method == 'standard' and not args.uif_reference_file):
        config.setdefault('evaluation', {})['require_uif_reference'] = False
    if args.seeds:
        import subprocess
        from scripts.run_multi_seed import aggregate_runs
        if len(args.seeds) != 5 or len(set(args.seeds)) != 5:
            raise ValueError('Aggregation requires exactly five distinct model seeds')
        aggregate = []
        for model_seed in args.seeds:
            command = [sys.executable, os.path.abspath(__file__), '--dataset', args.dataset,
                       '--backbone', args.backbone, '--method', args.method,
                       '--lambda1', str(args.lambda1), '--lambda2', str(args.lambda2),
                       '--eval_mode', args.eval_mode, '--loss_type', args.loss_type,
                       '--evaluation-stage', args.evaluation_stage,
                       '--config', args.config, '--seed', str(model_seed),
                       '--split-seed', str(args.split_seed), '--gpu', str(args.gpu),
                       '--save_dir', args.save_dir, '--results_dir', args.results_dir]
            if args.uif_reference_file:
                command.extend(['--uif-reference-file', args.uif_reference_file])
            if args.alignment_mode is not None:
                command.extend(['--alignment_mode', args.alignment_mode])
            if args.allow_missing_uif_reference:
                command.append('--allow-missing-uif-reference')
            subprocess.run(command, check=True)
            result_path = os.path.join(args.results_dir, args.dataset, args.backbone,
                                       args.method, f'seed_{model_seed}.json')
            with open(result_path, 'r') as handle:
                result = json.load(handle)
            aggregate.append(result)
        summary = aggregate_runs(aggregate)
        summary.update({
            'split_seed': args.split_seed,
            'selected_hyperparameters': {'lambda1': args.lambda1, 'lambda2': args.lambda2}})
        summary_path = os.path.join(args.results_dir, args.dataset, args.backbone,
                                    args.method, 'aggregate.json')
        with open(summary_path, 'w') as handle:
            json.dump(summary, handle, indent=2)
        print(f'Multi-seed summary saved to {summary_path}')
        return
    config.setdefault('seeds', {})['model'] = args.seed
    config['seeds']['data_split'] = args.split_seed
    device = get_device(args.gpu)
    print(f"Using device: {device}")

    ds_config = config.get('dataset', {}).get(args.dataset, {})
    seed_config = config.get('seeds', {})
    split_path = os.path.join(args.results_dir, 'splits',
                              f'{args.dataset}_seed{args.split_seed}.json')
    dataset = load_dataset(args.dataset,
                           min_ui=ds_config.get('min_user_interactions', 5),
                           min_ii=ds_config.get('min_item_interactions', 5),
                           data_split_seed=args.split_seed,
                           negative_sampling_seed=seed_config.get('negative_sampling', 42),
                           split_path=split_path)
    print(f"Dataset: {args.dataset}, Users: {dataset.n_users}, Items: {dataset.n_items}, "
          f"Interactions: {len(dataset.interactions)}")

    run_results_dir = os.path.join(args.results_dir, args.dataset, args.backbone, args.method)
    run_checkpoint_dir = os.path.join(args.save_dir, args.dataset, args.backbone, args.method)
    os.makedirs(run_checkpoint_dir, exist_ok=True)
    os.makedirs(run_results_dir, exist_ok=True)
    save_path = os.path.join(run_checkpoint_dir, f"seed_{args.seed}{args.output_suffix}.pt")

    if args.method == 'standard':
        backbone, user_embs, item_embs, ndcg, results = train_standard(
            dataset, args.backbone, config, device, args.eval_mode, args.loss_type,
            save_path, args.evaluation_stage)
    elif args.method == 'dual2fair':
        backbone, user_embs, item_embs, ndcg, results = train_dual2fair(
            dataset, args.backbone, config, device, args.lambda1, args.lambda2,
            args.eval_mode, args.loss_type, save_path, args.evaluation_stage)
    else:
        backbone, results = train_baseline(
            dataset, args.backbone, args.method, config, device, args.eval_mode, args.loss_type, save_path)

    results.update({
        'dataset': args.dataset,
        'backbone': args.backbone,
        'method': args.method,
        'model_seed': args.seed,
        'data_split_seed': args.split_seed,
        'negative_sampling_seed': seed_config.get('negative_sampling', 42),
        'ot_sampling_seed': seed_config.get('ot_sampling', 42),
        'evaluation_seed': seed_config.get('evaluation', 42),
        'split_hash': dataset.split_hash,
        'eval_mode': args.eval_mode,
        'evaluation_stage': args.evaluation_stage,
        'loss_type': args.loss_type,
        'alignment_mode': config.get('dual2fair', {}).get('alignment_mode'),
        'mmd_kernel': ('linear' if config.get('dual2fair', {}).get('alignment_mode') == 'mmd'
                       else None),
        'ablation_name': ({'hard': 'Hard Matching', 'mmd': 'MMD Alignment',
                           'ot': 'Dual2Fair'}.get(config.get('dual2fair', {}).get('alignment_mode'))
                          if args.method == 'dual2fair' else None),
        'selected_hyperparameters': {'lambda1': args.lambda1, 'lambda2': args.lambda2},
        'checkpoint_path': save_path,
        'baseline_processing_type': PROCESSING_TYPES.get(args.method),
    })
    if hasattr(backbone, '_best_validation_results'):
        results['validation_results'] = backbone._best_validation_results
    if hasattr(backbone, '_validation_trajectory'):
        results['validation_trajectory'] = backbone._validation_trajectory
    results_path = os.path.join(run_results_dir, f"seed_{args.seed}{args.output_suffix}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    metric_parts = [f"NDCG={results.get('NDCG', 0):.4f}",
                    f"Hit={results.get('Hit', 0):.4f}",
                    f"DUF={results.get('DUF', 0):.6f}",
                    f"DIF={results.get('DIF', 0):.6f}"]
    if results.get('UIF') is not None:
        metric_parts.append(f"UIF={results['UIF']:.6f}")
    print("\nFinal Results: " + ", ".join(metric_parts))
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    main()
