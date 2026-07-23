export CUDA_VISIBLE_DEVICES=0

model_name=SCNet
# d state 2
python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 862 \
  --patch_sizes 96 48 24 \
  --d_model 512 \
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --train_epochs 50\
  --d_ff 1024 \
  --itr 1 \
  --result_file result_long_term_forecast_traffic.txt \
  --batch_size 8

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 862 \
  --patch_sizes 96 48 24 \
  --d_model 512 \
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --train_epochs 50\
  --d_ff 1024 \
  --itr 1 \
  --result_file result_long_term_forecast_traffic.txt \
  --batch_size 8

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 862 \
  --patch_sizes 96 48 24 \
  --d_model 512 \
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --train_epochs 50\
  --d_ff 1024 \
  --itr 1 \
  --result_file result_long_term_forecast_traffic.txt \
  --batch_size 8

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 862 \
  --patch_sizes 96 48 24 \
  --d_model 512 \
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --train_epochs 50\
  --d_ff 1024 \
  --itr 1 \
  --result_file result_long_term_forecast_traffic.txt \
  --batch_size 8