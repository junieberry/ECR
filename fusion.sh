cd src_emo 
cp -r data/emo_data/* data/redial/
python data/redial/process.py 
accelerate launch train_pre.py \
--dataset redial \
--num_train_epochs 10 \
--gradient_accumulation_steps 4  \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 64  \
--num_warmup_steps 1389 \
--max_length 200 \
--prompt_max_length 200  \
--entity_max_length 32  \
--learning_rate 5e-4 \
--seed 42 \
--nei_mer
