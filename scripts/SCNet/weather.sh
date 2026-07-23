export CUDA_VISIBLE_DEVICES=0

model_name=SCNet

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 21 \
  --patch_sizes 96 48 24 12 6 3 \
  --d_model 256\
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 20\
  --d_ff 512\
  --itr 1 \
  --result_file result_long_term_forecast_weather.txt
  --dec_in 21 \
  --c_out 21 \

python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 21 \
  --patch_sizes 96 48 24 12 6 3 \
  --d_model 256\
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 20\
  --d_ff 512\
  --itr 1 \
  --result_file result_long_term_forecast_weather.txt


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 21 \
  --patch_sizes 96 48 24 12 6 3 \
  --d_model 256\
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 20\
  --d_ff 512\
  --itr 1 \
  --result_file result_long_term_forecast_weather.txt


python -u run.py \
  --is_training 1 \
  --task_name long_term_forecast \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 21 \
  --patch_sizes 96 48 24 12 6 3 \
  --d_model 256\
  --e_layers 2 \
  --des 'Exp' \
  --learning_rate 0.001 \
  --train_epochs 20\
  --d_ff 512\
  --itr 1 \
  --result_file result_long_term_forecast_weather.txt