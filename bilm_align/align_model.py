'''
Train and test bidirectional language models.
'''

import os
import sys
import time
import json
import re

import tensorflow as tf
import numpy as np

from tensorflow.python.ops.init_ops import glorot_uniform_initializer
# from tf.nn.rnn_cell import MultiRNNCell
from bilm_align.my_rnn import MultiRNNCell

from .data import Vocabulary, UnicodeCharsVocabulary, InvalidNumberOfCharacters

DTYPE = 'float32'
DTYPE_INT = 'int64'
PRINT_SHAPE = False

tf.logging.set_verbosity(tf.logging.INFO)

class LanguageModel(object):
    '''
    A class to build the tensorflow computational graph for NLMs

    All hyperparameters and model configuration is specified in a dictionary
    of 'options'.

    is_training is a boolean used to control behavior of dropout layers
        and softmax.  Set to False for testing.

    The LSTM cell is controlled by the 'lstm' key in options
    Here is an example:

     'lstm': {
      'cell_clip': 5,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 5,
      'projection_dim': 512,
      'use_skip_connections': True},

        'projection_dim' is assumed token embedding size and LSTM output size.
        'dim' is the hidden state size.
        Set 'dim' == 'projection_dim' to skip a projection layer.
    '''
    def __init__(self, options, is_training):
        self.options = options
        self.is_training = is_training
        self.bidirectional = options.get('bidirectional', False)

        # use word or char inputs?
        self.char_inputs = 'char_cnn' in self.options

        # for the loss function
        self.share_embedding_softmax = options.get(
            'share_embedding_softmax', False)
        if self.char_inputs and self.share_embedding_softmax:
            raise ValueError("Sharing softmax and embedding weights requires "
                             "word input")

        self.sample_softmax = options.get('sample_softmax', True)

        self._build()
        # sys.exit()

    def _build_word_embeddings(self):
        '''
        Builds the inputs and embedding
        Inputs: self.token_ids
        Outputs: self.embedding, self.embedding_reverse
        '''

        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # LSTM options
        projection_dim = self.options['lstm']['projection_dim']

        # the input token_ids and word embeddings
        self.token_ids = (
                            tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps),
                               name='token_ids'),
                            tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps),
                               name='token_ids'),
                           )
        # the word embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                "embedding", [n_tokens_vocab, projection_dim],
                dtype=DTYPE,
            )
            self.embedding = (
                                tf.nn.embedding_lookup(self.embedding_weights,
                                                self.token_ids[0]),
                                tf.nn.embedding_lookup(self.embedding_weights,
                                                self.token_ids[1])
                                )

        # if a bidirectional LM then make placeholders for reverse
        # model and embeddings
        if self.bidirectional:
            self.token_ids_reverse = (
                            tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps),
                               name='token_ids_reverse'),
                            tf.placeholder(DTYPE_INT,
                              shape=(batch_size, unroll_steps),
                              name='token_ids_reverse'),
                              )
            with tf.device("/cpu:0"):
                self.embedding_reverse = (
                    tf.nn.embedding_lookup(
                        self.embedding_weights, self.token_ids_reverse[0]),
                    tf.nn.embedding_lookup(
                        self.embedding_weights, self.token_ids_reverse[1])
                        )

        if PRINT_SHAPE:
            print("embedding.shape", self.embedding[0].get_shape())
            print("embedding_reverse.shape",
                self.embedding_reverse[0].get_shape())

    def _build_word_char_embeddings(self):
        '''
        options contains key 'char_cnn': {

        'n_characters': 262,

        # includes the start / end characters
        'max_characters_per_token': 50,

        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',

        # for the character embedding
        'embedding': {'dim': 16}

        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        Builds the inputs and embedding
        Inputs: self.tokens_characters
        Outputs: self.embedding, self.embedding_reverse
        '''
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']
        projection_dim = self.options['lstm']['projection_dim']

        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if n_chars != 261:
            raise InvalidNumberOfCharacters(
                    "Set n_characters=261 for training see the README.md"
            )
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the input character ids
        self.tokens_characters = (
            tf.placeholder(DTYPE_INT,
               shape=(batch_size, unroll_steps, max_chars),
               name='tokens_characters'),
            tf.placeholder(DTYPE_INT,
               shape=(batch_size, unroll_steps, max_chars),
               name='tokens_characters')
           )
        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                    "char_embed", [n_chars, char_embed_dim],
                    dtype=DTYPE,
                    initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = (
                tf.nn.embedding_lookup(self.embedding_weights,
                                    self.tokens_characters[0]),
                tf.nn.embedding_lookup(self.embedding_weights,
                                    self.tokens_characters[1])
                                    )

            if self.bidirectional:
                self.tokens_characters_reverse = (
                    tf.placeholder(DTYPE_INT,
                           shape=(batch_size, unroll_steps, max_chars),
                           name='tokens_characters_reverse'),
                    tf.placeholder(DTYPE_INT,
                           shape=(batch_size, unroll_steps, max_chars),
                           name='tokens_characters_reverse')
                   )
                self.char_embedding_reverse = (
                    tf.nn.embedding_lookup(self.embedding_weights,
                        self.tokens_characters_reverse[0]),
                    tf.nn.embedding_lookup(self.embedding_weights,
                        self.tokens_characters_reverse[1])
                    )

        # the convolutions
        def make_convolutions(inp, reuse):
            with tf.variable_scope('CNN', reuse=reuse) as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        #w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        #)

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                            inp, w,
                            strides=[1, 1, 1, 1],
                            padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                            conv, [1, 1, max_chars-width+1, 1],
                            [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)

        # for first model, this is False, for others it's True
        reuse = tf.get_variable_scope().reuse
        embedding = (
                make_convolutions(self.char_embedding[0], reuse),
                make_convolutions(self.char_embedding[1], reuse)
            )

        self.token_embedding_layers = [embedding]

        if self.bidirectional:
            # re-use the CNN weights from forward pass
            embedding_reverse = (
                    make_convolutions(self.char_embedding_reverse[0], True),
                    make_convolutions(self.char_embedding_reverse[1], True)
                )

        # for highway and projection layers:
        #   reshape from (batch_size, n_tokens, dim) to
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            embedding = (
                    tf.reshape(embedding[0], [-1, n_filters]),
                    tf.reshape(embedding[1], [-1, n_filters])
                )
            if self.bidirectional:
                embedding_reverse = (
                        tf.reshape(embedding_reverse[0], [-1, n_filters]),
                        tf.reshape(embedding_reverse[1], [-1, n_filters])
                    )

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                    W_proj_cnn = tf.get_variable(
                        "W_proj", [n_filters, projection_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                        dtype=DTYPE)
                    b_proj_cnn = tf.get_variable(
                        "b_proj", [projection_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

        # apply highways layers
        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

                embedding = (
                        high(embedding[0], W_carry, b_carry,
                            W_transform, b_transform),
                        high(embedding[1], W_carry, b_carry,
                            W_transform, b_transform)
                    )
                if self.bidirectional:
                    embedding_reverse = (
                        high(embedding_reverse[0], W_carry, b_carry,
                                W_transform, b_transform),
                        high(embedding_reverse[1], W_carry, b_carry,
                                W_transform, b_transform)
                    )
                self.token_embedding_layers.append(
                    (
                        tf.reshape(embedding[0],
                            [batch_size, unroll_steps, highway_dim]),
                        tf.reshape(embedding[1],
                            [batch_size, unroll_steps, highway_dim]),
                    )
                )

        # finally project down to projection dim if needed
        if use_proj:
            embedding = (
                tf.matmul(embedding[0], W_proj_cnn) + b_proj_cnn,
                tf.matmul(embedding[1], W_proj_cnn) + b_proj_cnn
            )
            if self.bidirectional:
                embedding_reverse = (
                    tf.matmul(embedding_reverse[0], W_proj_cnn) + b_proj_cnn,
                    tf.matmul(embedding_reverse[1], W_proj_cnn) + b_proj_cnn
                )
            self.token_embedding_layers.append(
                (tf.reshape(embedding[0],
                        [batch_size, unroll_steps, projection_dim]),
                 tf.reshape(embedding[1],
                        [batch_size, unroll_steps, projection_dim])
                )
            )

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = [batch_size, unroll_steps, projection_dim]
            embedding = (
                tf.reshape(embedding[0], shp), tf.reshape(embedding[1], shp)
                )
            if self.bidirectional:
                embedding_reverse = (
                    tf.reshape(embedding_reverse[0], shp),
                    tf.reshape(embedding_reverse[1], shp)
                    )

        # at last assign attributes for remainder of the model
        self.embedding = embedding
        if self.bidirectional:
            self.embedding_reverse = embedding_reverse

    def _build(self):
        '''
        Build the LSTM model
        Inputs:
        Outputs: lstm_outputs
        '''
        # size of input options
        n_tokens_vocab = self.options['n_tokens_vocab']
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        # LSTM options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        dropout = self.options['dropout']
        keep_prob = 1.0 - dropout

        self.n_lstm_layers = n_lstm_layers

        if self.char_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()

        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)
        self.init_lstm_state = []
        self.final_lstm_state = []

        # get the LSTM inputs
        if self.bidirectional:
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]

        # now compute the LSTM outputs
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')

        use_skip_connections = self.options['lstm'].get(
                                            'use_skip_connections')
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")

        # lstm_top_outputs = ([],[]) # the top-level lstm outputs
        lstm_outputs = (
                [[] for i in range(n_lstm_layers)],
                [[] for i in range(n_lstm_layers)])
        # the lstm outputs at every level lstm_outputs[0] for source
                                         #lstm_outputs[1] for target

        # iterate two directions
        for lstm_num, lstm_input in enumerate(lstm_inputs):
            # lstm_input[0]: source input
            # lstm_input[1]: target input
            lstm_cells = []
            # iterate layers
            for i in range(n_lstm_layers):
                # define LSTM
                if projection_dim < lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, num_proj=projection_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)

                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # add dropout
                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                        input_keep_prob=keep_prob)

                lstm_cells.append(lstm_cell)


            if n_lstm_layers > 1:
                lstm_cell = MultiRNNCell(lstm_cells)
            else:
                lstm_cell = lstm_cells[0]

            for k in [0, 1]: # iterate among source and target

                with tf.control_dependencies([lstm_input[k]]):
                    self.init_lstm_state.append(
                        lstm_cell.zero_state(batch_size, DTYPE))
                    # NOTE: this variable scope is for backward and
                    # source/target compatibility with existing models...
                    with tf.variable_scope('RNN_%s_%s' % (k, lstm_num)):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            lstm_cell,
                            tf.unstack(lstm_input[k], axis=1),
                            initial_state=self.init_lstm_state[-1])

                    self.final_lstm_state.append(final_state)

                if PRINT_SHAPE:
                    print("lstm_num", lstm_num)
                    print("len(_lstm_output_unpacked)",
                        len(_lstm_output_unpacked))
                    for tmp_i in range(len(_lstm_output_unpacked)):
                        print('i', tmp_i)
                        print("len(_lstm_output_unpacked[tmp_i])", len(_lstm_output_unpacked[tmp_i]))
                        for tmp_j, tmp_elm in enumerate(
                            _lstm_output_unpacked[tmp_i]):
                            print('j', tmp_j)
                            print("_lstm_output_unpacked.shape",
                                tmp_elm.get_shape())
                            break
                        break

                for i in range(n_lstm_layers):
                    # get the i-th layer outputs of lstm: unroll_steps *
                    # [(batch_size, projection_dim)]
                    _i_layer_output_unpacked = \
                        [t[i] for t in _lstm_output_unpacked]

                    # lstm_output_flat: (batch_size * unroll_steps, 512)
                    lstm_output_flat = tf.reshape(
                        tf.stack(_i_layer_output_unpacked, axis=1),
                        [-1, projection_dim])
                    if self.is_training:
                        # add dropout to output
                        lstm_output_flat = tf.nn.dropout(lstm_output_flat,
                            keep_prob)
                    if i == n_lstm_layers-1:
                        tf.add_to_collection('lstm_output_embeddings',
                            _i_layer_output_unpacked)

                    if PRINT_SHAPE:
                        print("lstm_output_flat.shape",
                            lstm_output_flat.get_shape())
                    lstm_outputs[k][i].append(lstm_output_flat)

        self._build_loss(lstm_outputs)

    def _build_loss(self, lstm_outputs):
        '''
        lstm_outputs:
            lstm_outputs[0]: source outputs
            lstm_outputs[1]: target outputs

        Create:
            self.total_loss: total loss op for training
            self.softmax_W, softmax_b: the softmax variables
            self.next_token_id / _reverse: placeholders for gold input

        '''
        batch_size = self.options['batch_size']
        unroll_steps = self.options['unroll_steps']

        n_tokens_vocab = self.options['n_tokens_vocab']

        # DEFINE next_token_id and *_reverse placeholders for the gold input
        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps),
                                   name=name)
            return id_placeholder

        # get the window and weight placeholders
        self.next_token_id = (
            _get_next_token_placeholders('source'),
            _get_next_token_placeholders('target')
            )
        if self.bidirectional:
            self.next_token_id_reverse = (
                _get_next_token_placeholders('source_reverse'),
                _get_next_token_placeholders('target_reverse'),
                )

        # DEFINE THE SOFTMAX VARIABLES
        # get the dimension of the softmax weights
        # softmax dimension is the size of the output projection_dim
        softmax_dim = self.options['lstm']['projection_dim']

        # the output softmax variables -- they are shared if bidirectional
        if self.share_embedding_softmax:
            # softmax_W is just the embedding layer
            self.softmax_W = self.embedding_weights

        with tf.variable_scope('softmax'), tf.device('/cpu:0'):
            # Glorit init (std=(1.0 / sqrt(fan_in))
            softmax_init = tf.random_normal_initializer(0.0,
                1.0 / np.sqrt(softmax_dim))
            if not self.share_embedding_softmax:
                self.softmax_W = tf.get_variable(
                    'W', [n_tokens_vocab, softmax_dim],
                    dtype=DTYPE,
                    initializer=softmax_init
                )
            self.softmax_b = tf.get_variable(
                'b', [n_tokens_vocab],
                dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

        # now calculate losses
        # loss for each direction of the LSTM
        self.individual_losses = []

        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        else:
            next_ids = [self.next_token_id]

        # log-likelihoood loss
        for k in [0, 1]: # iterate over source and target
            top_lstm_outpus = lstm_outputs[k][-1]
            for next_ids_tuple, lstm_output_flat in \
                zip(next_ids, top_lstm_outpus): # iterate over forw/rev

                # flatten the LSTM output and next token id gold to shape:
                # (batch_size * unroll_steps, softmax_dim)
                # Flatten and reshape the token_id placeholders
                id_placeholder = next_ids_tuple[k]
                next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])

                with tf.control_dependencies([lstm_output_flat]):
                    if self.is_training and self.sample_softmax:
                        losses = tf.nn.sampled_softmax_loss(
                                       self.softmax_W, self.softmax_b,
                                       next_token_id_flat, lstm_output_flat,
                                       self.options['n_negative_samples_batch'],
                                       self.options['n_tokens_vocab'],
                                       num_true=1)

                    else:
                        # get the full softmax loss
                        output_scores = tf.matmul(
                            lstm_output_flat,
                            tf.transpose(self.softmax_W)
                        ) + self.softmax_b
                        # NOTE: tf.nn.sparse_softmax_cross_entropy_with_logits
                        #   expects unnormalized output since it performs the
                        #   softmax internally
                        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                            logits=output_scores,
                            labels=tf.squeeze(next_token_id_flat, squeeze_dims=[1])
                        )

                self.individual_losses.append(tf.reduce_mean(losses))

        # now make the total loss -- it's the mean of the individual losses
        if self.bidirectional:
            self.total_loss = 0.25 * (self.individual_losses[0]
                + self.individual_losses[1] + self.individual_losses[2]
                + self.individual_losses[3])
        else:
            self.total_loss = 0.5 * (self.individual_losses[0]
                + self.individual_losses[1])

    def _build_align_loss(self, lstm_outputs):
        '''
        lstm_outputs:
            lstm_outputs[0]: source outputs
            lstm_outputs[1]: target outputs
        '''
        align_losses = []

        for i in self.n_lstm_layers:
            for src, trg in zip(lstm_outputs[0][i], lstm_outputs[1][i]):
                # build loss for src and trg:
                # (batch_size * unroll_steps, projection_dim)

                align_losses.append(loss)

        return  tf.add_n(align_losses)/len(align_losses)
