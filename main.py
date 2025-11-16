import argparse
import time
import os
import glob
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from omegaconf import OmegaConf

from model import FeedForwardPredictor, FeedForwardPredictorParams
from data import CaidaData
from config import Config, TrainingParams, ModelConfig
from sketch import CountMinSketch


def parse_command_line_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--key-column", type=str, required=True)
    parser.add_argument("--value-column", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--config-file", type=str, default="config.yml", required=True)
    args = parser.parse_args()
    return args


def train(args: argparse.Namespace):
    paths = glob.glob(args.input_path)
    data = CaidaData(paths=paths, key_col=args.key_column, value_col=args.value_column)

    # Load and validate config parameters.
    structured_config = OmegaConf.structured(Config)
    raw_config = OmegaConf.load(args.config_file)
    init_config = OmegaConf.merge(structured_config, raw_config)
    config: Config = OmegaConf.to_object(init_config)

    # Extract training and model config.
    train_params = config.training_params
    model_config = config.model_config

    device = train_params.device if torch.cuda.is_available() else 'cpu'

    dataloader = DataLoader(data, batch_size=train_params.batch_size, shuffle=train_params.shuffle, collate_fn=lambda x: x)

    model = FeedForwardPredictor(model_config.params).to(device=device)

    optimizer = AdamW(model.parameters())

    best_loss = float('inf')
    for epoch in range(train_params.epochs):
        print(f'Epoch: {epoch}')
        avg_loss = 0
        for i, (keys, X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            keys = keys.to(device)
            X = X.to(device=device).type(torch.float32)
            y = y.to(device=device)
            output = model(X=X, keys=keys)
            loss = (((output - y) / (y + 1e-8)) ** 2).mean()
            loss.backward()

            avg_loss += (loss - avg_loss) / (i + 1)

            if i % train_params.logging_frequency == 0:
                print(f'Average Batch Loss ({i + 1}/{len(dataloader)}): {avg_loss}')
                if avg_loss < best_loss:
                    best_loss = avg_loss.item()
                    model.save(args.model_save_path)
                    print(f'Saved best model with loss {best_loss} to {args.model_save_path}')

            optimizer.step()

        if avg_loss < best_loss:
            best_loss = avg_loss.item() # type: ignore
            model.save(args.model_save_path)
            print(f'Saved best model with loss {best_loss} to {args.model_save_path}')

    # Save final model state.
    filename = os.path.splitext(os.path.basename(args.model_save_path))[0]
    filedir = os.path.dirname(args.model_save_path)
    final_model_path = os.path.join(filedir, f'{filename}_final.pth')
    model.save(final_model_path)
    print(f'Saved final model to {final_model_path}')


if __name__ == '__main__':
    args = parse_command_line_args()
    train(args)