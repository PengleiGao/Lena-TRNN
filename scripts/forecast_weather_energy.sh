export CUDA_VISIBLE_DEVICES=0

model_name=lena_rnn

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --d_layers 2 \
  --enc_in 21 \
  --c_out 21 \
  --d_model 128 \
  --score_cof 10 \
  --num_step 2 \
  --learning_rate 0.001 \
  --batch_size 128 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --d_layers 2 \
  --enc_in 21 \
  --c_out 21 \
  --d_model 128 \
  --score_cof 10 \
  --num_step 2 \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --des 'Exp' \
  --itr 1 \


python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --d_layers 2 \
  --enc_in 21 \
  --c_out 21 \
  --d_model 128 \
  --score_cof 20 \
  --num_step 2 \
  --learning_rate 0.0001 \
  --batch_size 64 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --d_layers 2 \
  --enc_in 21 \
  --c_out 21 \
  --d_model 128 \
  --score_cof 20 \
  --num_step 2 \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --des 'Exp' \
  --itr 1 \