python -m torch.distributed.launch --nproc_per_node 8 train.py --config_name "visdrone2021" --gpus "0,1,2,3,4,5,6,7"
# python -m torch.distributed.launch --nproc_per_node 6 train.py --config_name "ab1_mot17_model1_20frames" --gpus "0,1,2,3,4,5"
