seq_len=512
model=CVC

for gpt_layer in 6
do
for pred_len in 96 192 336 720
do

python runfew.py \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh2_$seq_len'_'$pred_len \
    --data ETTh2 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.001 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layers $gpt_layer \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 15 \
    --percent 10 \
    --output_w 1 \
    --feature_w 0.1

echo '====================================================================================================================='
done
done
