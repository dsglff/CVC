seq_len=100
model=CVC


for pred_len in 24 36 48 60
do
for percent in 100
do

python runh2.py \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --is_training 1 \
    --task_name long_term_forecast \
    --model_id illness_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --lradj type1 \
    --train_epochs 100 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
    --enc_in 7 \
    --c_out 7 \
    --gpt_layers 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --r 8 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --patience 10 \
    --percent $percent \
    --prompt_length 3 \
    --task_loss smooth_l1 \
    --output_loss mase \
    --output_loss smooth_l1

done
done

