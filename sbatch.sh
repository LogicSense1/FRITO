#!/bin/bash
#SBATCH -J passt                        # 作业名
#SBATCH -o slurm-%j-passt.high_low_branch.out                       # 屏幕上的输出文件重定向到 slurm-%j.out , %j 会替换成jobid
#SBATCH -e slurm-%j-passt.high_low_branch.err                       # 屏幕上的错误输出文件重定向到 slurm-%j.err , %j 会替换成jobid
#SBATCH -p compute                              # 作业提交的分区为 compute
#SBATCH -N 1                                    # 作业申请 1 个节点
#SBATCH -t 12:00:00                            # 任务运行的最长时间为 1 小时
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:2

pwd; hostname;
TIME=$(date -Iseconds)
echo $TIME

LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/anaconda3/envs/ba3l/lib DDP=2 CUDA_VISIBLE_DEVICES=0,1 python ex_nsynth.py with models.net.rf_norm_t=high_low_branch trainer.use_tensorboard_logger=True -p --debug