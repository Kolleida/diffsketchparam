import argparse
import sys
import os
import glob
import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from tqdm import tqdm
from loguru import logger


from model import FeedForwardPredictor, FeedForwardPredictorParams
from data import CaidaData
from config import Config, TrainingParams, ModelConfig
from sketch import CountMinSketch


def parse_command_line_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--input-path", type=str, nargs="+", required=True)
    parser.add_argument("--num-sketches", type=int, default=25)
    parser.add_argument("--key-column", type=str, required=True)
    parser.add_argument("--value-column", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--config-file", type=str, default="config.yml", required=True)
    parser.add_argument("--log-level", type=str, default='INFO')
    parser.add_argument("--use-entropy", action='store_true', default=False)
    args = parser.parse_args()
    return args


def train(args: argparse.Namespace):
    paths = []
    for path in args.input_path:
        paths += glob.glob(path)
    data = CaidaData(paths=paths, key_col=args.key_column, value_col=args.value_column, num_sketches=args.num_sketches)

    # Load and validate config parameters.
    config: Config = Config.from_yaml(args.config_file)

    # Extract training and model config.
    train_params = config.training_params
    model_config = config.model_config

    device = train_params.device if torch.cuda.is_available() else 'cpu'

    dataloader = DataLoader(data, batch_size=train_params.batch_size, shuffle=train_params.shuffle, collate_fn=lambda x: x)

    model = FeedForwardPredictor(model_config.params).to(device=device)

    if args.use_entropy:
        entropy_predictor = torch.nn.Sequential(
            torch.nn.Linear(2, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device=device)

    optimizer_params = train_params.optimizer_parms
    if optimizer_params.optimizer_type == "AdamW":
        optimizer = AdamW(model.parameters(), **optimizer_params.optimizer_options)
    elif optimizer_params.optimizer_type == "SGD":
        optimizer = SGD(model.parameters(), **optimizer_params.optimizer_options)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_params.optimizer_type}")
    if optimizer_params.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_params.epochs * len(dataloader))

    best_loss = float('inf')
    for epoch in range(train_params.epochs):
        logger.info(f'Epoch: {epoch}')
        avg_loss = 0
        for i, (keys, X, y) in enumerate(dataloader):
            optimizer.zero_grad()
            
            keys = keys.to(device)
            X = X.to(device=device).type(torch.float32)
            y = y.to(device=device)
            output = model(X=X, keys=keys)
            rel_loss = (((output[:, :-1] - y[:, :-1]) / (y[:, :-1] + 1e-8)) ** 2).mean()

            entropy_loss = 0.0
            if args.use_entropy:
                entropy_pred = f.softplus(entropy_predictor(output[:, :-1]).reshape(-1, 1))
                entropy_loss = (((entropy_pred - y[:, :-1]) / (y[:, :-1] + 1e-8)) ** 2).mean()

            loss = rel_loss + entropy_loss * 1e-2
            loss.backward()

            avg_loss += (rel_loss - avg_loss) / (i + 1)

            if i % train_params.logging_frequency == 0:
                logger.info(f'Average Batch Loss ({i + 1}/{len(dataloader)}): {avg_loss}')
                if avg_loss < best_loss:
                    best_loss = avg_loss.item()
                    model.save(args.model_save_path)
                    logger.info(f'Saved best model with loss {best_loss} to {args.model_save_path}')

            optimizer.step()
            if optimizer_params.use_scheduler:
                scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss.item() # type: ignore
            model.save(args.model_save_path)
            logger.info(f'Saved best model with loss {best_loss} to {args.model_save_path}')

    # Save final model state.
    filename = os.path.splitext(os.path.basename(args.model_save_path))[0]
    filedir = os.path.dirname(args.model_save_path)
    final_model_path = os.path.join(filedir, f'{filename}_final.pth')
    model.save(final_model_path)
    logger.info(f'Saved final model to {final_model_path}')


if __name__ == '__main__':
    args = parse_command_line_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    train(args)