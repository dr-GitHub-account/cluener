time=$(date "+%Y%m%d%H%M%S")
output_time=${time}

CUDA_VISIBLE_DEVICES=1 python run_lstm_crf.py \
    --do_eval \
    --batch_size 64 \
    --output_time ${output_time}