"""General utilities for argument parser."""
import argparse


def str2bool(v):
    """Converts string to boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def default_argument_parser():
    """Default argument parser."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data_root', default='/mnt/local/datasets/manual/', type=str,
        help='path to the directory (default: /mnt/local/datasets/manual/)')
    parser.add_argument(
        '--data_name', default='imagenet2012', type=str,
        choices=['imagenet2012', 'domainnet'])

    parser.add_argument(
        '--clip_model', default='openai/clip-vit-base-patch16', type=str,
        help='https://huggingface.co/models?filter=clip')
    parser.add_argument(
        '--clip_cls_init', default=None, type=str,
        help='how to initialize cls_params (default: None)')
    parser.add_argument(
        '--clip_ext_init', default=None, type=str,
        help='how to initialize ext_params (default: None)')

    parser.add_argument(
        '--batch_size', default=256, type=int,
        help='the number of examples for each mini-batch (default: 256)')
    parser.add_argument(
        '--num_workers', default=32, type=int,
        help='how many subprocesses to use for data loading (default: 32)')
    parser.add_argument(
        '--prefetch_factor', default=2, type=int,
        help='number of batches loaded in advance by each worker (default: 2)')

    return parser
