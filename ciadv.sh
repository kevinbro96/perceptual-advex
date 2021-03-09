#!/bin/bash
#SBATCH -A test                  # 自己所属的账户 (不要改)
#SBATCH -J ciattack               # 所运行的任务名称 (自己取)
#SBATCH -N 1                    # 占用的节点数（根据代码要求确定）
#SBATCH --ntasks-per-node=1     # 运行的进程数（根据代码要求确定）
#SBATCH --cpus-per-task=8       # 每个进程的CPU核数 （根据代码要求确定）
#SBATCH --gres=gpu:1            # 占用的GPU卡数 （根据代码要求确定）
#SBATCH -p gpu                  # 任务运行所在的分区 (根据代码要求确定，gpu为gpu分区，gpu4为4卡gpu分区，cpu为cpu分区)
#SBATCH -t 1-12:00:00            # 运行的最长时间 day-hour:minute:second
#SBATCH -o attack.out        # 打印输出的文件
conda activate attack
python adv_train.py --batch_size 128 --arch resnet50 --dataset cifar --dataset_path ../data --attack "CIAttack(model, '../auto_aug-master/results/emb64_1_5/model_new.pth')" --only_attack_correct
