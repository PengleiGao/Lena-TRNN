export CUDA_VISIBLE_DEVICES=1

model_name=lena_rnn

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --d_layers 4 \
  --d_model 256 \
  --score_cof 20 \
  --num_step 3 \
  --enc_in 21 \
  --c_out 21 \
  --batch_size 128 \
  --itr 1 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.25 \
  --mask_rate 0.25 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --d_layers 4 \
  --d_model 256 \
  --score_cof 20 \
  --num_step 3 \
  --enc_in 21 \
  --c_out 21 \
  --batch_size 128 \
  --itr 1 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.375 \
  --mask_rate 0.375 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --d_layers 4 \
  --d_model 256 \
  --score_cof 20 \
  --num_step 3 \
  --enc_in 21 \
  --c_out 21 \
  --batch_size 128 \
  --itr 1 \
  --learning_rate 0.001

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_mask_0.5 \
  --mask_rate 0.5 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --d_layers 4 \
  --d_model 256 \
  --score_cof 20 \
  --num_step 3 \
  --enc_in 21 \
  --c_out 21 \
  --batch_size 128 \
  --itr 1 \
  --learning_rate 0.001
