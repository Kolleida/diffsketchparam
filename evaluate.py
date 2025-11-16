from sketch import CountMinSketch
from model import *
from data import CaidaData
from config import Config
import polars as pl
import torch
from torch.utils.data import DataLoader
import time
import glob
import argparse


def parse_command_line_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--key-column", type=str, required=True)
    parser.add_argument("--value-column", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    args = parser.parse_args()
    return args


def eval_model(args: argparse.Namespace):
    paths = glob.glob(args.input_path)
    data = CaidaData(paths=paths, key_col=args.key_column, value_col=args.value_column)

    dataloader = DataLoader(data, batch_size=16384, shuffle=True, collate_fn=lambda x: x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FeedForwardPredictor.load(path=args.model_save_path).to(device=device)
    model.eval()

    avg_loss = 0
    total_loss = 0
    n = 0
    with torch.inference_mode():
        for i, (keys, X, y) in enumerate(dataloader):
            keys = keys.to(device)
            X = X.to(device=device).type(torch.float32)
            y = y.to(device=device)
            output = model(X=X, keys=keys)

            clamped_output = torch.clamp(torch.round(output), min=1.0)

            n += y.shape[0]
            loss = (((clamped_output - y) / (y + 1e-8)) ** 2).sum()
            total_loss += loss.item()

            avg_loss = total_loss / n

            if i % 400 == 0:
                print(f'Average Relative Error ({i + 1}/{len(dataloader)}): {avg_loss}')


if __name__ == "__main__":
    args = parse_command_line_args()
    start_time = time.time()
    eval_model(args)
    end_time = time.time()
    print(f"Evaluation completed in {end_time - start_time} seconds.")

