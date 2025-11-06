import os
import polars as pl
import polars_hash as plh
import numpy as np


class CountMinSketch:
    # TODO: Maybe make this use lazy frames instead, so we can parallelize updates across files?

    KEY_COL = "keys"
    VALUE_COL = "values"

    def __init__(self, d: int, w: int) -> None:
        self.d = d # Number of rows (hash functions).
        self.w = w # Number of columns.
        self.table = np.zeros((d, w))

        rnd = os.urandom
        self.index_seeds = [int.from_bytes(rnd(4), "little") for _ in range(self.d)]

    def insert(self, data: pl.DataFrame, key_col: str = "flows", value_col: str | None = None):
        df = data.select(
            pl.col(key_col).alias(self.KEY_COL), 
            (pl.col(value_col) if value_col is not None else pl.lit(1)).alias(self.VALUE_COL)
        )
        hashed_indices = self._hash_keys(df)
        for d in range(self.d):
            idx_col = f'{self.KEY_COL}_{d}'
            col_df = pl.DataFrame([hashed_indices.get_column(idx_col), df.get_column(self.VALUE_COL)])

            agg_updates = col_df.group_by(pl.col(idx_col)).sum()
            indices = agg_updates.get_column(idx_col).to_numpy()
            increments = agg_updates.get_column(self.VALUE_COL).to_numpy()
            self.table[d, indices] += increments
        
    def _hash_col(self, keys: pl.Series, d=0) -> np.ndarray:
        if not (0 <= d < self.d):
            raise ValueError("Row index out of range.")
        hashed_vals = keys.hash(seed=d)
        indices = (hashed_vals % self.w).to_numpy()
        return indices
    
    def _hash_keys(self, data: pl.DataFrame | pl.Series) -> pl.DataFrame:
        """
        Takes a DataFrame/Series of keys and creates additional columns with indices for each
        hash function, named like KEY_0, KEY_1, KEY_2, etc.
        """
        if isinstance(data, pl.Series):
            data = data.to_frame()
        hashed_cols = [
            pl.col(self.KEY_COL).
            hash(seed=d).
            mod(self.w).
            alias(f"{self.KEY_COL}_{d}") 
            for d in range(self.d)
        ]
        result = data.select(*hashed_cols)
        return result

    def query(self, keys: pl.Series | pl.DataFrame, key_col: str = "packets") -> np.ndarray:
        keys = keys.alias(self.KEY_COL).to_frame() if isinstance(keys, pl.Series) else keys.select(pl.col(key_col).alias(self.KEY_COL))
        all_values = []
        hashed_indices = self._hash_keys(keys)
        for d in range(self.d):
            idx_col = f'{self.KEY_COL}_{d}'
            indices = hashed_indices.get_column(idx_col).to_numpy()
            entries = self.table[d, indices]
            all_values.append(entries)
        freqs_matrix = np.stack(all_values)
        freqs = np.min(freqs_matrix, axis=0)
        return freqs

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Tried to add object of type {type(other)} to {self.__class__.__name__}")
        if self.table.shape != other.table.shape:
            raise ValueError("Dimensions of underlying table do not match.")
        result = self.__class__(d=self.d, w=self.w)
        result.table += self.table + other.table
        return result
    

class CountSketch:

    def __init__(self, d: int, w: int) -> None:
        self.d = d # Number of rows (hash functions).
        self.w = w # Number of columns.
        self.table = np.zeros((d, w))

    def insert(self, keys: pl.Series, updates: pl.Series | None = None):
        if updates is None:
            updates = pl.Series("values", [1] * len(keys))
        
        row_indices = np.arange(self.d)
        for d in row_indices:
            col_indices, signs = self._hash_indices(keys, d=int(d))
            df = pl.DataFrame(dict(indices=col_indices, updates=updates * signs))
            agg_updates = df.group_by(pl.col("indices")).sum()
            indices = agg_updates["indices"].to_numpy()
            increments = agg_updates["updates"].to_numpy()
            self.table[d, indices] += increments
        
    def _hash_indices(self, keys: pl.Series, d=0) -> tuple[np.ndarray, np.ndarray]:
        """Each key gets mapped to a bucket index, along with its corresponding sign."""
        if not (0 <= d < self.d):
            raise ValueError("Row index out of range.")
        indices = (keys.hash(seed=d) % self.w).to_numpy()
        signs = (keys.hash(seed=d, seed_1=d) % 2).to_numpy().astype(int)
        signs[signs == 0] = -1
        return indices, signs
    
    def query(self, keys):
        all_values = []
        row_indices = np.arange(self.d)
        for d in row_indices:
            col_indices, signs = self._hash_indices(keys, d=int(d))
            entries = self.table[d, col_indices] * signs
            all_values.append(entries)
        freqs_matrix = np.stack(all_values)
        freqs = np.median(freqs_matrix, axis=0)
        return freqs

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Tried to add object of type {type(other)} to {self.__class__.__name__}")
        if self.table.shape != other.table.shape:
            raise ValueError("Dimensions of underlying table do not match.")
        result = self.__class__(d=self.d, w=self.w)
        result.table += self.table + other.table
        return result