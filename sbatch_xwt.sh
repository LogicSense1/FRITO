#!/bin/bash
#SBATCH -J passt                        # 作业名
#SBATCH -o slurm-%j-passt.out                       # 屏幕上的输出文件重定向到 slurm-%j.out , %j 会替换成jobid
#SBATCH -e slurm-%j-passt.err                       # 屏幕上的错误输出文件重定向到 slurm-%j.err , %j 会替换成jobid
#SBATCH -p gpu                              # 作业提交的分区为 compute
#SBATCH -t 24:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12

pwd; hostname;
TIME=$(date -Iseconds)
echo $TIME

# source ~/.bashrc
# conda activate ba3l

# .01 .05 .10 .25 .50 .75
# LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/ba3l/lib CUDA_VISIBLE_DEVICES=0 python ex_dcase20_dev.py with models.net.rf_norm_t=attn_penalty models.net.global_tokens_num=2 models.net.attention_penalty_width=297 models.net.stripe_width=99 models.net.att_penalty_scale=16.0 trainer.use_tensorboard_logger=True -p --debug
# python ex_dcase20_dev.py with models.net.rf_norm_t=per_row_8 trainer.use_tensorboard_logger=True -p --debug
DDP=4 python ex_dcase20_dev.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug
DDP=4 python ex_dcase20_dev.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug
DDP=4 python ex_dcase20_dev.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug
DDP=4 python ex_dcase20_dev.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug


# DDP=4 python ex_nsynth.py with models.net.rf_norm_t=high_low trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with models.net.rf_norm_t=high_low trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with models.net.rf_norm_t=high_low trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with models.net.rf_norm_t=high_low trainer.use_tensorboard_logger=True -p --debug
 
# DDP=4 python ex_nsynth.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with models.net.s_patchout_t=10 models.net.s_patchout_f=5 trainer.use_tensorboard_logger=True -p --debug

# DDP=4 python ex_nsynth.py with trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with trainer.use_tensorboard_logger=True -p --debug
# DDP=4 python ex_nsynth.py with trainer.use_tensorboard_logger=True -p --debug
