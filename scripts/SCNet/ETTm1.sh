export CUDA_VISIBLE_DEVICES=0

model_name=SCNet
# d state 2
python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --patch_sizes 96 48 24 \
  --d_model 256  \
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.0003 \
  --train_epochs 20\
  --d_ff 512 \
  --itr 1 \
  --batch_size 32 \
  --result_file result_long_term_forecast_ETTm1.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 192  \
  --enc_in 7 \
  --patch_sizes 96 72 48 \
  --d_model 384 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --train_epochs 40\
  --d_ff 1024 \
  --itr 1 \
  --result_file result_long_term_forecast_ETTm1.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 7 \
  --patch_sizes 96 72 48 \
  --d_model 384 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --train_epochs 20\
  --d_ff 1024 \
  --itr 1 \
  --result_file result_long_term_forecast_ETTm1.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720 \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --patch_sizes 96 72 48 \
  --d_model 384 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.0001 \
  --train_epochs 20\
  --d_ff 1024 \
  --itr 1 \
  --result_file result_long_term_forecast_ETTm1.txt