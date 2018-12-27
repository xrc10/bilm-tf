
import argparse

import numpy as np

from bilm_align.training_align import train, load_vocab
from bilm_align.data import BidirectionalLMDataset

def count_tokens(fname):
    '''
    counts the tokens given the file name
    '''
    num_words = 0

    with open(fname, 'r') as f:
        for line in f:
            words = line.split()
            num_words += len(words)
    return num_words

def main(args):
    # load the vocab
    src_vocab = load_vocab(args.src_vocab_file, None)
    trg_vocab = load_vocab(args.trg_vocab_file, None, prefix_n=src_vocab.size)

    # define the options
    batch_size = 64  # batch size for each GPU
    n_gpus = 1

    # number of tokens in training data (this for 1B Word Benchmark)
    # n_train_tokens = args.n_train_tokens
    n_train_tokens = 0
    for prefix in [args.src_train_prefix, args.trg_train_prefix]:
        print('Counting tokens in {}'.format(args.src_train_prefix))
        n_train_tokens += count_tokens(args.src_train_prefix)
        print('Total tokens {}'.format(n_train_tokens))

    options = {
     'bidirectional': True,

     # 'char_cnn': {'activation': 'tanh',
     #  'embedding': {'dim': 4},
     #  'filters': [
     #      [1, 8],
     #      [2, 8],
     #      [3, 16],
     #      [4, 32],
     #      [5, 64],
     #  ],
     #  'max_characters_per_token': 50,
     #  'n_characters': 261,
     #  'n_highway': 1},

     'dropout': 0.1,

     'lstm': {
      'cell_clip': 3,
      'dim': 256,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 64,
      'use_skip_connections': True},

     'all_clip_norm_val': 10.0,

     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': src_vocab.size + trg_vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 2048,
    }

    src_prefix = args.src_train_prefix
    trg_prefix = args.trg_train_prefix

    src_data = BidirectionalLMDataset(src_prefix, src_vocab, test=False,
                                      shuffle_on_load=True)
    trg_data = BidirectionalLMDataset(trg_prefix, trg_vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir

    train(options, src_data, trg_data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--src_model_dir', help='Source model file')
    parser.add_argument('--trg_model_dir', help='Target model file')
    parser.add_argument('--src_vocab_file', help='Source vocabulary file')
    parser.add_argument('--trg_vocab_file', help='Target vocabulary file')
    parser.add_argument('--src_train_prefix', help='Source prefix for train files')
    parser.add_argument('--trg_train_prefix', help='Target prefix for train files')
    # parser.add_argument('--alignment', help='alignment file')

    args = parser.parse_args()
    main(args)
