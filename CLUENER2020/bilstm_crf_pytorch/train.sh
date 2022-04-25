time=$(date "+%Y%m%d%H%M%S")
output_time=${time}

CUDA_VISIBLE_DEVICES=1 python run_lstm_crf.py \
    --do_train \
    --batch_size 64 \
    --epochs 50 \
    --output_time ${output_time}