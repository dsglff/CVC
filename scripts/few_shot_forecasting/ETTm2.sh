seq_len=512
model=CVC

percent=10

for gpt_layer in 6
do
for pred_len in 96 192 336 720
do

python runfew.py \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id ETTm2_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data ETTm2 \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --train_epochs 10 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --percent $percent \
    --gpt_layer $gpt_layer \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 20 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 5 \
    --feature_w 0.01 \
    --output_w 1 \
    --task_loss l1 \
    --output_loss smooth_l1 \
    --feature_loss l1 \

echo '====================================================================================================================='
done
done