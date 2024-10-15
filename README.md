# LogLM

<p align="center">
    <img src="asset/LogLM.png" width=600 />
</p>

This repo contains training codes and the human-calibriated instruction dataset for LogLM, an LLM designed for instruction-based log analysis.

Paper: [LogLM: From Task-based to Instruction-based Automated Log Analysis](https://arxiv.org/abs/2410.09352)

## 📣 Introduction
<p align="center">
    <img src="asset/illustration.png" width=800 />
</p>

Automatic log analysis is essential for the efficient Operation and Maintenance (O\&M) of software systems, providing critical insights into system behaviors. However, existing approaches mostly treat log analysis as training a model to perform an isolated task, using task-specific log-label pairs. These task-based approaches are inflexible in generalizing to complex scenarios, depend on task-specific training data, and cost significantly when deploying multiple models. In this paper, we propose an instruction-based training approach that transforms log-label pairs from multiple tasks and domains into a unified format of instruction-response pairs. Our trained model, LogLM, can follow complex user instructions and generalize better across different tasks, thereby increasing flexibility and reducing the dependence on task-specific training data. By integrating major log analysis tasks into a single model, our approach also relieves model deployment burden. Experimentally, LogLM outperforms existing approaches across five log analysis capabilities, and exhibits strong generalization abilities on complex instructions and unseen tasks. 

<p align="center">
    <img src="asset/performance.png"  width=400 />
</p>

<p align="center">
    <img src="asset/case_study.png"  width=1000 />
</p>

## 🔰 Installation
```
$ pip install requirements.txt
```
## ✨ Instruction Dataset of LogLM
We created an instruction dataset of 2632 pairs of instruction and response for training LogLM, an instruction-following LLM. The instruction dataset consists of five log analysis tasks and nine distinct log domains.

Example:
```
Instruction Dataset of LogLM.json
{
  "Instruction": "Interpret the error message in the given log.",
  "Input": "ValueError: Unable to configure handler 'log_file': [Errno 2] No such file or directory: '/logs/django.log'",
  "Response": "The error message indicates that the Django application's logging system is configured to write logs to a file located at '/logs/django.log'. However, it encountered a ValueError because it could not find the specified file or directory. This could be due to an incorrect path or the directory not existing where it expects to find or create the log file."
}
```
The statistics of the dataset is as follows:
<p align="center">
    <img src="asset/instruction dataset.png"  width=500 />
</p>

## 📝 Training LogLM

LogLM in our paper is implemented by fine-tuning LLaMA-2-7B on our curated instruction dataset. Thanks [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for implementing an efficient tool to fine-tune LLMs. The training steps of LogLM are as follows:


(1) Prepare the environment


Please make sure to install the packages required. For issues related to environment, please refer to LLaMA-Factory v0.3.2.

(2) Register the training dataset


Register the dataset in data/dataset_info.json.  
```
dataset_info.json
{
  "LogLM_train": {
    "file_name": "Instruction Dataset of LogLM.json",
    "file_sha1": "",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
```

(3) Start training (with multiple GPUs)
```
WANDB_MODE=offline deepspeed --include localhost:0,1 --master_port=9905 src/train_bash.py \
    --deepspeed ds_config_zero2.json \
    --stage sft \
    --model_name_or_path [path to Llama-2-7b-hf] \
    --do_train \
    --dataset LogLM_train \
    --template alpaca \
    --finetuning_type full \
    --output_dir [path to model saving] \
    --overwrite_cache \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --save_strategy "epoch" \
    --save_total_limit 6 \
    --logging_steps 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --num_train_epochs 6 \
    --plot_loss \
    --overwrite_output_dir \
    --bf16 True \
    --tf32 True \
```
The config file ds_config_zero2.json can be found in [DeepSpeed](https://github.com/huggingface/transformers/blob/main/tests/deepspeed/ds_config_zero2.json)

(4) Inference


The inference test datasets in the "test set" folder should also be registerd in data/dataset_info.json, the same way as the training dataset. 
```
python src/train_bash.py \
    --stage sft \
    --model_name_or_path [path to Llama-2-7b-hf] \
    --do_predict \
    --dataset [registered name of your test set] \
    --template alpaca \
    --finetuning_type full \
    --checkpoint_dir [path to finetuned model] \
    --output_dir [result saving path] \
    --per_device_eval_batch_size 32 \
    --predict_with_generate
```
For more information, please refer to LLaMA-Factory v0.3.2.
