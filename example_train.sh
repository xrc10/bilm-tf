# 1 Prepare input data and a vocabulary file.
# See data/

export CUDA_VISIBLE_DEVICES=$(python ../../tools/test_cuda/test_cuda.py 4000 2>&1)
echo "Training on GPU $CUDA_VISIBLE_DEVICES"
mkdir -p exp/;

# 2 Train the biLM.
# python bin/train_elmo_small.py \
#     --train_prefix='data/train/data.*' \
#     --vocab_file data/vocab.txt \
#     --save_dir exp &> exp/train.log.txt;

# 3. Evaluate the trained model.

# 4. Convert the tensorflow checkpoint to hdf5 for prediction with bilm or allennlp.
# python bin/dump_weights.py \
#     --save_dir exp \
#     --outfile exp/weights.hdf5 | tee exp/dump.log.txt;
