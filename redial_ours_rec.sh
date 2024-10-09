# merge infer results from conversation model of UniCRS
device=$1
data=redial_ours
seed=$2

seed=2024

cd src_emo/data/${data}

python merge.py ${data} ${seed}

cd ../..


accelerate launch train_rec.py   \
    --dataset ${redial_ours} \
    --n_prefix_rec 10    \
    --num_train_epochs 5   \
    --per_device_train_batch_size 16   \
    --per_device_eval_batch_size 32   \
    --gradient_accumulation_steps 8   \
    --num_warmup_steps 530   \
    --context_max_length 200   \
    --prompt_max_length 200   \
    --entity_max_length 32   \
    --learning_rate 1e-4   \
    --seed 8   \
    --like_score 2.0   \
    --dislike_score 1.0   \
    --notsay_score 0.5    \
    --weighted_loss   \
    --nei_mer  \
    --use_sentiment \
    --output_dir data/saved/${data}_rec_${seed}   \
    --project ecr   \
    --name ${data}_rec_${seed}   \
    --seed ${seed}   \
    --use_wandb
     
