#!/usr/bin/env python3
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


import argparse
from frpn.analysis import md_diagnostics as module


def main():
    parser = argparse.ArgumentParser(description="Reproduce the manuscript's 1417 structure-control matched pairs from archived OOF predictions.")
    parser.add_argument('--raw-csv', default='data/raw/md_final1640_v2/final_1640dataset.csv')
    parser.add_argument('--fold-csv', default='data/splits/md_final1640_v2/with_fold.csv')
    parser.add_argument('--predictions', default='results/oof_predictions/md_final1640_v2')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    RAW_CSV, FOLD_CSV, PREDS, OUT = map(Path, [args.raw_csv, args.fold_csv, args.predictions, args.output])
    OUT.mkdir(parents=True, exist_ok=True)


    raw = pd.read_csv(RAW_CSV).copy()
    raw.insert(0, 'row_idx', np.arange(len(raw), dtype=np.int64))
    meta, comp_vec, _, _ = module._prepare_meta(RAW_CSV, FOLD_CSV, OUT / 'meta')
    stats = module._temp_stats(raw, module.LABELS)

    rows = []
    # Exact temperature and chain length are controlled. Chemistry profile, rigidity,
    # and polarity metadata are intentionally not matching variables.
    for (temperature, chain_len), group in meta.groupby(['glob_feat0', 'chain_len'], dropna=False):
        indices = group['row_idx'].to_numpy(dtype=np.int64)
        if len(indices) < 2:
            continue
        overlaps = module._pairwise_overlaps(comp_vec, indices, indices)
        left_local, right_local = np.triu_indices(len(indices), k=1)
        # Preserve the float32 boundary behavior used by the original analysis.
        chemistry_match = overlaps[left_local, right_local] >= float(np.float32(0.95))
        for a, b in zip(left_local[chemistry_match], right_local[chemistry_match]):
            left = int(indices[a])
            right = int(indices[b])
            topology_changed = str(meta.at[left, 'topology']) != str(meta.at[right, 'topology'])
            ordering_changed = str(meta.at[left, 'mix_mode']) != str(meta.at[right, 'mix_mode'])
            if not (topology_changed or ordering_changed):
                continue
            rows.append({
                'match_type': 'chemistry_matched_structure_variation',
                'glob_feat0': float(temperature),
                'chain_len': int(chain_len),
                'row_i': left,
                'row_j': right,
                'overlap': float(overlaps[a, b]),
                'topology_i': str(meta.at[left, 'topology']),
                'topology_j': str(meta.at[right, 'topology']),
                'ordering_i': str(meta.at[left, 'mix_mode']),
                'ordering_j': str(meta.at[right, 'mix_mode']),
            })

    pairs = pd.DataFrame(rows)
    assert len(pairs) > 0
    pairs.to_csv(OUT / 'structure_variation_pairs.csv', index=False)

    metrics = module._pair_metrics(
        pairs, raw=raw, preds_dir=PREDS, stats=stats,
        control_name='chemistry_matched_structure_variation')
    metrics.to_csv(OUT / 'structure_variation_delta_metrics_by_label.csv', index=False)
    summary = module._pair_metric_summary(metrics)
    summary.to_csv(OUT / 'structure_variation_delta_metrics_summary.csv', index=False)

    matrix = metrics.pivot_table(
        index='model', columns='label', values='corr_delta', aggfunc='mean'
    ).reindex(index=module.MODELS, columns=module.LABELS)
    module._plot_heatmap(
        matrix,
        OUT / 'topology_control_corr_4models_revised.png',
        OUT / 'topology_control_corr_4models_revised.pdf',
        title='Chemistry-matched structure variation: corr(Δŷ_z, Δy_z)',
        cmap=LinearSegmentedColormap.from_list(
            'topology_blue',
            ['#DCEBFA', '#AECBE2', '#7FA8CC', '#4F739D', '#254468', '#0B1B33'],
            N=256),
        sequential=True)

    usage = np.bincount(
        np.concatenate([pairs['row_i'].to_numpy(dtype=np.int64),
                        pairs['row_j'].to_numpy(dtype=np.int64)]),
        minlength=len(raw))
    mean_corr = metrics.groupby('model')['corr_delta'].mean().to_dict()
    per_label = {}
    controls = ['Uni_Macro', 'Stage2_Only']
    for label in module.LABELS:
        subset = metrics[metrics['label'] == label].set_index('model')
        frpn = float(subset.at['FRPN', 'corr_delta'])
        control_values = {model: float(subset.at[model, 'corr_delta']) for model in controls}
        per_label[label] = {
            'FRPN_corr_delta': frpn,
            **{f'{model}_corr_delta': value for model, value in control_values.items()},
            'FRPN_minus_best_control': frpn - max(control_values.values()),
            'FRPN_best_among_FRPN_and_two_controls': bool(
                frpn >= max([frpn] + list(control_values.values()))),
        }

    result = {
        'protocol': {
            'chemistry_overlap_minimum': 0.95,
            'temperature': 'exactly matched',
            'chain_length': 'exactly matched',
            'structural_difference': 'topology or sequence ordering differs',
            'not_controlled': ['chem_profile_id', 'rigid_ratio', 'polar_ratio'],
        },
        'n_pairs': int(len(pairs)),
        'unique_samples': int((usage > 0).sum()),
        'maximum_pairs_per_sample': int(usage.max()),
        'mean_corr_delta_by_model': {key: float(value) for key, value in mean_corr.items()},
        'per_label_comparison': per_label,
        'frpn_best_label_count_among_frpn_and_two_controls': int(sum(
            item['FRPN_best_among_FRPN_and_two_controls'] for item in per_label.values())),
        'additive_null_and_stacking_recomputed': False,
        'reason_additive_null_and_stacking_unchanged': (
            'These analyses use all OOF records and do not consume the matched-pair set.'),
    }
    (OUT / 'summary.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
