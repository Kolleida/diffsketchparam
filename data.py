import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
import numpy as np
from itertools import product
import math

from sketch import CountMinSketch, CountSketch


eps_int = (1e-6, 1e-2)
delta_int = (1e-3, 1e-2)


class CaidaData(Dataset):

    KEY_COL = 'flow'
    VALUE_COL = 'packets'

    def __init__(self, paths: str | list[str], key_col: str = 'flow', value_col: str = 'packets') -> None:
        super().__init__()
        self.paths = paths
        self.df = pl.scan_csv(paths).select(
            pl.col(key_col).alias(self.KEY_COL), 
            pl.col(value_col).alias(self.VALUE_COL)
        ).collect()
        self.sketches = self.initialize_sketches()
        self.true_counts = self.df.group_by(self.KEY_COL).sum()
        self.create_dataset()

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

    def create_dataset(self):
        keys = self.true_counts.select(self.KEY_COL)
        per_sketch = []
        for cm in self.sketches:
            approx_freqs = cm.query(keys, key_col=self.KEY_COL)
            true_freqs = self.true_counts.get_column(self.VALUE_COL).to_numpy()
            diffs = np.abs(approx_freqs - true_freqs)
            # Each data point should be: [flow key embedding], approx freq, abs error, true d, true w.
            # TODO: How to embed flow key. Maybe use hash embeddings (HashingVectorizer)? Or use bits of hash value? Also, normalize abs error and freq based on stream length?
            entry = np.stack([approx_freqs, diffs, np.full(len(keys), cm.d), np.full(len(keys), cm.w)], axis=1)
            per_sketch.append(entry)
        return per_sketch

    def __getitem__(self, index) -> torch.Tensor:
        return super().__getitem__(index)