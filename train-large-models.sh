wget https://raw.githubusercontent.com/sberbank-ai/ru-gpts/master/pretrain_transformers.py
wget http://files.deeppavlov.ai/datasets/sber_squad-v1.1.tar.gz

python prepare_training_data.py

python pretrain_transformers.py     --output_dir=answer_model_large     --model_type=gpt2     --model_name_or_path=sberbank-ai/rugpt3large_based_on_gpt2     --do_train     --train_data_file=train2.txt     --per_gpu_train_batch_size 1     --gradient_accumulation_steps 4     --num_train_epochs 2     --block_size 1024     --overwrite_cache     --overwrite_output_dir     --save_steps 5000     --fp16

python pretrain_transformers.py     --output_dir=question_model_large     --model_type=gpt2     --model_name_or_path=sberbank-ai/rugpt3large_based_on_gpt2     --do_train     --train_data_file=train.txt     --per_gpu_train_batch_size 1     --gradient_accumulation_steps 4     --num_train_epochs 2     --block_size 1024     --overwrite_cache     --overwrite_output_dir     --save_steps 5000     --fp16


