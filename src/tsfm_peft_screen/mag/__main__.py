"""CPU inference using an exported fixed MAG bundle and observed inputs only."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .model import load_bundle

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--input', required=True, help='NPZ with observed [B,512] and sigma [B], both float32')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Output already exists; refusing to overwrite it')
    torch.set_num_threads(4)
    data = np.load(args.input, allow_pickle=False)
    if set(data.files) != {'observed', 'sigma'}:
        raise ValueError('Only observed and TRAIN sigma may appear in the input NPZ')
    model = load_bundle(args.snapshot, args.bundle, device='cpu')
    with torch.inference_mode():
        result = model(torch.from_numpy(data['observed']), torch.from_numpy(data['sigma']))
    with output.open('xb') as f:
        np.save(f, result.numpy(), allow_pickle=False)
    print(f'Saved {tuple(result.shape)}; CPU inference only, optimizer updates=0')
if __name__ == '__main__':
    main()
