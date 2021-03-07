

python adv_train.py --batch_size 128 --arch resnet50 --dataset cifar --dataset_path ../data --attack "CIAttack(model, '../auto_aug-master/results/emb64_301/model_new.pth')" --only_attack_correct
