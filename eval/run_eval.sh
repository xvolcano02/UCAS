model_dirs=()
device=1
template="our"
temperatures="0.6"
max_tokens=8000
N_SAMPLING=8
top_p=1.0
use_chat=1
use_system_prompt=1
#task="aime25,aime24,math500,amc23,minerva_math,olympiadbench"
task="aime24"
echo "Model directories:"
for model_dir in "${model_dirs[@]}"; do
    echo "${model_dir}"
done
seeds="42"
for model_dir in "${model_dirs[@]}"; do
    CUDA_VISIBLE_DEVICES=${device} python eval_baseline.py \
        --model_name ${model_dir} \
        --template ${template} \
        --seeds "${seeds}" \
        --temperatures "${temperatures}" \
        --top_p "${top_p}" \
        --save True
done