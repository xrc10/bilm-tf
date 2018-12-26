#!/usr/bin/env python

"""Test bilm_align/data.py.
"""

from __future__ import print_function
import os
import sys
import argparse

from bilm_align.data import Vocabulary, LMDataset


def main(arguments):
    vocab_file = \
        '/usr1/home/ruochenx/research/Elmo/data/UM/raw/dict.bilm.en.txt'
    vocab = Vocabulary(vocab_file, validate_file=True)

    dataset_file = \
        '/usr1/home/ruochenx/research/Elmo/data/UM/raw/en.sent.txt.small'
    lm_dataset = LMDataset(dataset_file, vocab, shuffle_on_load=True)


    for sent in lm_dataset.get_sentence():
        print(sent)
        break

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
