export CUDA_VISIBLE_DEVICES=0

model_name=SCNet
# d state 2
python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --patch_sizes 96 48 \
  --d_model 384 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.00001 \
  --train_epochs 20\
  --d_ff 1536 \
  --itr 1 \
  --result_file result_long_term_forecast_ETTm2.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 7 \
  --patch_sizes 96 48 36 24 12 6\
  --d_model 256 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --train_epochs 20\
  --d_ff 256 \
  --itr 1 \
  --result_file result_long_term_forecast_ETTm2.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 7 \
  --patch_sizes 96 48 \
  --d_model 384 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.00001 \
  --train_epochs 20\
  --d_ff 1536 \
  --itr 1 \
  --result_file result_long_term_forecast_ETTm2.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720 \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --patch_sizes 96 48 \
  --d_model 384 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.00001 \
  --train_epochs 20\
  --d_ff 1536 \
  --itr 1 \
  --result_file result_long_term_forecast_ETTm2.txt