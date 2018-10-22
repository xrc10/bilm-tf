
import argparse

import numpy as np

from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 10)

    # define the options
    batch_size = 2  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 134

    options = {
     'bidirectional': True,

     'char_cnn': {'activation': 'tanh',
      'embedding': {'dim': 4},
      'filters': [
          [1, 8],
          [2, 8],
          [3, 16],
          [4, 32],
          [5, 64],
      ],
      'max_characters_per_token': 10,
      'n_characters': 261,
      'n_highway': 1},

     'dropout': 0.1,

     'lstm': {
      'cell_clip': 3,
      'dim': 16,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 8,
      'use_skip_connections': True},

     'all_clip_norm_val': 1.0,

     'n_epochs': 50,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 10,
     'n_negative_samples_batch': 16,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')

    args = parser.parse_args()
    main(args)
