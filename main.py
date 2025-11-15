import argparse
import time
import os
import glob
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from model import SketchParameterPredictor, HashEmbeddingConfig
from data import CaidaData
from sketch import CountMinSketch


def parse_command_line_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--key-column", type=str, required=True)
    parser.add_argument("--value-column", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, default="sketch_model.pth")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args


def train(args: argparse.Namespace):
    device = args.device if torch.cuda.is_available() else 'cpu'

    paths = glob.glob(args.input_path)
    data = CaidaData(paths=paths, key_col=args.key_column, value_col=args.value_column)
    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)

    hash_config = HashEmbeddingConfig()
    model = SketchParameterPredictor(
        input_dim=2, 
        hidden_dim=1024,
        output_dim=2,
        num_hidden_layers=8,
        hash_embedding_config=hash_config
    ).to(device=device)

    optimizer = AdamW(model.parameters())

    best_loss = float('inf')
    for epoch in range(args.epochs):
        print(f'Epoch: {epoch}')
        avg_loss = 0
        for i, (keys, X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            keys = keys.to(args.device)
            X = X.to(device=args.device).type(torch.float32)
            y = y.to(device=args.device)
            output = model(X=X, keys=keys)
            loss = (((output - y) / (y + 1e-8)) ** 2).mean()
            loss.backward()

            avg_loss += (loss - avg_loss) / (i + 1)

            if i % 400 == 0:
                print(f'Average Batch Loss ({i + 1}/{len(dataloader)}): {avg_loss}')
                if avg_loss < best_loss:
                    best_loss = avg_loss.item()
                    torch.save(model.state_dict(), args.model_save_path)
                    print(f'Saved best model with loss {best_loss} to {args.model_save_path}')

            optimizer.step()

        if avg_loss < best_loss:
            best_loss = avg_loss.item() # type: ignore
            torch.save(model.state_dict(), args.model_save_path)
            print(f'Saved best model with loss {best_loss} to {args.model_save_path}')

    # Save final model state.
    filename = os.path.splitext(os.path.basename(args.model_save_path))[0]
    filedir = os.path.dirname(args.model_save_path)
    final_model_path = os.path.join(filedir, f'{filename}_final.pth')
    torch.save(model.state_dict(), os.path.join(filedir, final_model_path))
    print(f'Saved final model to {os.path.join(filedir, final_model_path)}')


if __name__ == '__main__':
    args = parse_command_line_args()
    train(args)