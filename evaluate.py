from sketch import CountMinSketch
from model import *
from data import CaidaData
from config import Config
import polars as pl
import torch
import numpy as np
from torch.utils.data import DataLoader
from loguru import logger
import time
import sys
import glob
import argparse
import json


def parse_command_line_args():
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--key-column", type=str, required=True)
    parser.add_argument("--num-sketches-dataset", type=int, default=25)
    parser.add_argument("--num-sketches-test", type=int, default=25)
    parser.add_argument("--value-column", type=str, required=True)
    parser.add_argument("--model-save-path", type=str, required=True)
    parser.add_argument("--results-path", type=str, default='results.json')
    parser.add_argument("--eval-type", type=str, choices=['dataset', 'test_sketches'], default='test_sketches')
    parser.add_argument("--log-level", type=str, default='INFO')
    args = parser.parse_args()
    return args


def evaluate_model_against_dataset(args: argparse.Namespace):
    paths = glob.glob(args.input_path)
    data = CaidaData(paths=paths, key_col=args.key_column, value_col=args.value_column, num_sketches=args.num_sketches_dataset)

    dataloader = DataLoader(data, batch_size=16384, shuffle=False, collate_fn=lambda x: x)

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
    print(f'Final Average Relative Error: {avg_loss}')

def evaluate_model_sketches(args: argparse.Namespace):
    paths = glob.glob(args.input_path)
    data = CaidaData(paths=paths, key_col=args.key_column, value_col=args.value_column, num_sketches=args.num_sketches_dataset)

    dataloader = DataLoader(data, batch_size=16384, shuffle=False, collate_fn=lambda x: x)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = FeedForwardPredictor.load(path=args.model_save_path).to(device=device)
    model.eval()

    eps_int = (1e-5, 1e-2)

    rng = np.random.default_rng()

    epsilons = np.exp(rng.uniform(np.log(eps_int[0]), np.log(eps_int[1]), size=args.num_sketches_test))

    average_errors = np.zeros(len(epsilons))
    relative_errors_to_eps = np.zeros(len(epsilons))
    cm_params = []
    num_undershoot = 0
    for eps_i, eps in enumerate(epsilons):
        logger.info(f'Evaluating for target epsilon: {eps:.4e} (sketch {eps_i + 1})')

        # Model ingests keys from the stream and target epsilon, outputs (d, w) sketch parameters. Average predictions across batch to get an estimate.
        # The average of all estimates over all batches is the final estimate.
        estimates = []
        with torch.inference_mode():
            for i, (keys, X, y) in enumerate(dataloader):
                keys = keys.to(device)
                X = X.to(device=device).type(torch.float32)

                X[:, 1] = eps # Want to get a sketch size with target epsilon.

                output = model(X=X, keys=keys)

                clamped_output = torch.clamp(torch.round(output), min=1.0)
                estimates.append(clamped_output.mean(dim=0))

                if i % 400 == 0:
                    logger.debug(f'Estimates after {i + 1}/{len(dataloader)} batches: {torch.stack(estimates).mean(dim=0)}')
                    
        final_estimate = torch.stack(estimates).mean(dim=0)
        logger.info(f'Final estimate: d = {final_estimate[0]}, w = {final_estimate[1]}')

        # Evaluate on the sketch with the estimated parameters.
        d, w = int(final_estimate[0]), int(final_estimate[1])
        cm_params.append([d, w])
        cm = CountMinSketch(d=int(final_estimate[0]), w=int(final_estimate[1]))
        cm.insert(data.df, key_col=data.KEY_COL, value_col=data.VALUE_COL)

        # Get relative and absolute error wrt target epsilon.
        keys = data.true_counts.select(data.KEY_COL)
        approx_freqs = cm.query(keys, key_col=data.KEY_COL) / data.stream_length
        true_freqs = data.true_counts.get_column(data.VALUE_COL).to_numpy() / data.stream_length
        diffs = np.abs(approx_freqs - true_freqs)

        avg_error = np.mean(diffs) # Error is normalized based on stream length.
        rel_error_to_eps = (avg_error - eps) / eps
        # Want to track whether our error goal was met.
        if avg_error - eps <= 0:
            num_undershoot += 1
        logger.info(f'Average Error of Estimated Sketch: {avg_error:.4e} ({avg_error - eps:+.4e} from epsilon {eps:.4e})')
        logger.info(f'Relative Error of Observed Error to Target Epsilon: {rel_error_to_eps:.4f} ')
        logger.info(f'Proportion of Target Errors Met: {num_undershoot / (eps_i + 1):.4f}')

        average_errors[eps_i] = avg_error
        relative_errors_to_eps[eps_i] = rel_error_to_eps

    logger.info(f'Final Average Error: {average_errors.mean():.4e}')
    logger.info(f'Final Relative Error: {relative_errors_to_eps.mean():.4f} (signed), {np.abs(relative_errors_to_eps).mean():.4f} (absolute)')

    results = {
        'num_input_sketches': args.num_sketches_dataset,
        'sketch_params': cm_params,
        'epsilons': epsilons.tolist(),
        'average_errors': average_errors.tolist(),
        'relative_errors': relative_errors_to_eps.tolist(),
        'num_undershoot': num_undershoot
    }
    try:
        with open(args.results_path, mode='w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Computed evaluation results saved to {args.results_path}')
    except Exception as e:
        logger.warning(f'Failed to save results: {e}')


if __name__ == "__main__":
    args = parse_command_line_args()
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)
    start_time = time.time()
    if args.eval_type == 'dataset':
        evaluate_model_against_dataset(args)
    elif args.eval_type == 'test_sketches':
        evaluate_model_sketches(args)
    else:
        logger.error(f'Unsupported evaluation type: {args.eval_type}')
    end_time = time.time()
    logger.info(f"Evaluation completed in {end_time - start_time} seconds.")

