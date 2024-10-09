device=$1
data=redial
seed=2024

CONDA_ENV_PATH="./.conda"

# Check if the conda environment exists
if [ -d "$CONDA_ENV_PATH" ]; then
    echo "Activating Conda environment from $CONDA_ENV_PATH..."
    
    source "$CONDA_ENV_PATH/bin/activate" base
fi

cd src_emo 
CUDA_VISIBLE_DEVICES=${device} WANDB_API_KEY=1364bc3808835e00258645d1d878f3a919d3d4a2 accelerate launch train_pre.py \
--dataset ${data} \
--num_train_epochs 10 \
--gradient_accumulation_steps 4  \
--per_device_train_batch_size 32 \
--per_device_eval_batch_size 128  \
--num_warmup_steps 1389 \
--max_length 200 \
--prompt_max_length 200  \
--entity_max_length 32  \
--learning_rate 5e-4 \
--seed 42 \
--nei_mer \
--project ecr \
--name ${data}_pre_${seed} \
--seed ${seed} \
# --use_wandb \
