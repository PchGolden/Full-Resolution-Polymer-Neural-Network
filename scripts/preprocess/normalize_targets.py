"""MD benchmark label transforms, matching src_MD/main.py.

python normalize_targets.py --input md_dataset_with_split_identifiers.csv --fold 0 --output fold0_targets.csv --parameters fold0_norm.json
Statistics use training rows only (fold != held-out fold), separately for each target.
"""
import argparse
import json
import numpy as np
import pandas as pd

LABELS = ['density', 'Rg', 'D', 'S_q_peak', 'nematic_order', 'dielectric_constant', 'refractive_index']

def fit_transform(frame, fold):
    y = frame[LABELS].to_numpy(dtype=np.float64).copy()
    if not np.isfinite(y).all():
        raise ValueError('Labels must be finite')
    y[:, LABELS.index('D')] = np.log10(np.clip(y[:, LABELS.index('D')], 1e-16, None))
    training = y[frame['fold'].to_numpy() != fold]
    if not len(training):
        raise ValueError('Empty training partition')
    mean = training.mean(axis=0)
    std = training.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    parameters = dict(label_names=LABELS, log10_labels=['D'], d_clamp_min=1e-16,
                      mean=mean.tolist(), std=std.tolist())
    return (y - mean) / std, parameters

def inverse_transform(z, parameters):
    y = np.asarray(z, dtype=np.float64) * np.asarray(parameters['std']) + np.asarray(parameters['mean'])
    y[..., LABELS.index('D')] = np.power(10.0, y[..., LABELS.index('D')])
    return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parameters', required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    z, parameters = fit_transform(frame, args.fold)
    result = pd.DataFrame(z, columns=[label + '_z' for label in LABELS])
    result.insert(0, 'fold', frame['fold'])
    result.insert(0, 'dataset_record', np.arange(1, len(frame) + 1))
    result.to_csv(args.output, index=False)
    with open(args.parameters, 'w') as handle:
        json.dump(parameters, handle, indent=2)
