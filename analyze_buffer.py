import numpy as np
import sys

base = sys.argv[1] if len(sys.argv) > 1 else '/tmp/buffer_dumps/accel_21x21_sfl_baseline/0'

dumps = {
    '10k': f'{base}/buffer_dump_10k.npz',
    '20k': f'{base}/buffer_dump_20k.npz',
    '30k': f'{base}/buffer_dump_30k.npz',
    'final': f'{base}/buffer_dump_final.npz',
}

for label, path in dumps.items():
    try:
        data = np.load(path)
    except FileNotFoundError:
        print(f'{label}: FILE NOT FOUND')
        continue

    size = int(data['size'])
    tokens = data['tokens'][:size]
    scores = data['scores'][:size]

    unique = len(np.unique(tokens, axis=0))

    rng = np.random.default_rng(42)
    idx = rng.choice(len(tokens), min(500, len(tokens)), replace=False)
    sample = tokens[idx].astype(np.float32)
    n = len(sample)
    hamming_sum = 0
    count = 0
    for i in range(n):
        diffs = (sample[i] != sample[i+1:]).mean(axis=1)
        hamming_sum += diffs.sum()
        count += len(diffs)
    hamming = hamming_sum / count

    print(f'\n=== {label} ===')
    print(f'  Buffer size:    {size}')
    print(f'  Unique levels:  {unique} ({100*unique/size:.1f}%)')
    print(f'  Hamming dist:   {hamming:.4f}')
    print(f'  Score mean/std: {scores.mean():.3f} / {scores.std():.3f}')
