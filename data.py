import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import polars_hash as plh
import numpy as np
from itertools import product
import math

from sketch import CountMinSketch, CountSketch


eps_int = (1e-6, 1e-2)
delta_int = (1e-3, 1e-2)


class CaidaData(Dataset):

    KEY_COL = 'flow'
    VALUE_COL = 'packets'

    def __init__(self, paths: str | list[str], key_col: str = 'flow', value_col: str = 'packets', dtype = torch.float32) -> None:
        super().__init__()
        self.paths = paths
        self.df = pl.scan_csv(paths).select(
            pl.col(key_col).alias(self.KEY_COL), 
            pl.col(value_col).alias(self.VALUE_COL)
        ).collect()
        self.sketches = self.initialize_sketches()
        self.true_counts = self.df.group_by(self.KEY_COL).sum()
        self.stream_length = self.true_counts.select(self.VALUE_COL).sum().item()
        self.data = self.create_dataset()

    def initialize_sketches(self) -> list[CountMinSketch]:
        sketches = []
        rng = np.random.default_rng()

        # TODO: Is there a better way to range over possible configs?
        errors = rng.uniform(eps_int[0], eps_int[1], size=5)
        deltas = rng.uniform(delta_int[0], delta_int[1], size=5)
        errors = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
        deltas = [1e-4, 1e-3, 1e-2]

        for eps, delta in product(errors, deltas):
            w = math.ceil(2 / eps)
            d = math.ceil(math.log(1 / delta, 2))
            print(f"w = {w}, d = {d} with eps = {eps}, delta = {delta}")
            cm = CountMinSketch(d=d, w=w)
            cm.insert(self.df, key_col=self.KEY_COL, value_col=self.VALUE_COL)
            sketches.append(cm)
        return sketches

    def create_dataset(self, dtype=torch.float32):
        keys = self.true_counts.select(self.KEY_COL)
        # Hash flow keys to torch.int64 indices (treat as dictionary). Need to treat these separately since HashEmbeddings require LongTensors.
        key_idx = keys.select(
            plh.col(self.KEY_COL).nchash.
            wyhash().
            mod(pl.Int64.max())
        ).to_torch(dtype=pl.Int64).flatten()
        float_data = []
        ground_truth_data = []
        for cm in self.sketches:
            # Normalize frequencies by stream length.
            approx_freqs = cm.query(keys, key_col=self.KEY_COL) / self.stream_length
            true_freqs = self.true_counts.get_column(self.VALUE_COL).to_numpy() / self.stream_length
            diffs = np.abs(approx_freqs - true_freqs)

            freq_info = torch.from_numpy(np.stack([approx_freqs, diffs], axis=1)).to(dtype=dtype)
            d_vals = torch.full((len(keys),), cm.d)
            w_vals = torch.full((len(keys),), cm.w)
            ground_truth = torch.stack([d_vals, w_vals], dim=-1).to(dtype=dtype)

            float_data.append(freq_info)
            ground_truth_data.append(ground_truth)

        flattened_key_indices = key_idx.repeat(len(self.sketches)).unsqueeze(dim=-1)
        flattened_float_data = torch.concat(float_data)
        flattened_gt_data = torch.concat(ground_truth_data)

        return flattened_key_indices, flattened_float_data, flattened_gt_data

    def __getitem__(self, index):
        key_indices, numeric_data, gt = self.data
        return key_indices[index], numeric_data[index], gt[index]
    
    def __len__(self):
        return len(self.data[0])