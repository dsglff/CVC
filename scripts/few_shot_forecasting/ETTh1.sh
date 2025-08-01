

seq_len=512
model=CVC

for gpt_layer in 6
do
for pred_len in 336 
do

python runfew.py \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTh1_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ETTh1 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 8 \
    --learning_rate 0.0005 \
    --lradj type1 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layer $gpt_layer \
    --itr 1 \
    --model $model \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 50 \
    --percent 10 \
    --output_w 1 \
    --feature_w 0.01 \
    --task_loss l1 \
    --output_loss smooth_l1 \
    --feature_loss l1 \

echo '====================================================================================================================='
done
done