export CUDA_VISIBLE_DEVICES=0

model_name=SCNet
# d state 2
python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_96 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 137 \
  --patch_sizes 96 48 24 12 \
  --d_model 256 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 20\
  --d_ff 512 \
  --itr 1 \
  --result_file result_long_term_forecast_solar.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_192 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 137 \
  --patch_sizes 96 48 24 12 \
  --d_model 256 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 20\
  --d_ff 512 \
  --itr 1 \
  --result_file result_long_term_forecast_solar.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_336 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 137 \
  --patch_sizes 96 48 24 12 \
  --d_model 256 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 20\
  --d_ff 512 \
  --itr 1 \
  --result_file result_long_term_forecast_solar.txt

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id Solar_96_720 \
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 137 \
  --patch_sizes 96 48 24 12 \
  --d_model 256 \
  --e_layers 1 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 20\
  --d_ff 512 \
  --itr 1 \
  --result_file result_long_term_forecast_solar.txt