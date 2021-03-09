#!/bin/bash
# SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J mywork               # 所运行的任务名称 (自己取)
# SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=10       # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1            # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p gpu                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 1-12:00:00            # 运行的最长时间 day-hour:minute:second
#SBATCH -o l2attack.out        # 打印输出的文件

conda activate attack                # 激活的虚拟环境名称
# 运行代码
 python adv_train.py --batch_size 128 --arch resnet50 --dataset cifar --attack "L2Attack(model, dataset_name='cifar',num_iterations=7)" --only_attack_correct --dataset_path ~/data --eps 1.0
