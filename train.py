from email import message
import os
from pathlib import Path
import shutil
from yacs.config import CfgNode as CN
import time
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.backends import cudnn

from src.data.dataset import MOTTrainDataset
from src.data.data_loader import collate_fn_dets
from src.models.build_model_loss import build_model_loss 
from src.optimizers import build_optimizer, build_scheduler
from src.utils.log import epoch_log

def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True 

def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param

def setDir(filepath):
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)

def main(args):
    init_seeds(0)

    default_yaml = './src/config/train.yaml'
    default_cfg = open(default_yaml)
    cfg = CN.load_cfg(default_cfg)
    if args.config_name!='default':
        cfg.merge_from_file(os.path.join(args.exp_dir, args.config_name,'train.yaml'))
    cfg.freeze()
    
    # store ckpts
    output_dir = Path(os.path.join(args.ckpt_dir, args.config_name))
    output_dir.mkdir(exist_ok=True, parents=True)
    log_path = os.path.join(output_dir, "log.txt")
    log = open(log_path,'w')

    # 记录config参数配置
    print(cfg, file=log, flush=True)
    # log.write(cfg + '\n')

    # ############################ DDP准备 ###################################
    local_rank = int(args.local_rank)
    torch.cuda.set_device(local_rank) # 根据local_rank来设定当前使用哪块GPU
    dist.init_process_group(backend='nccl') # 初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端
    device = torch.device("cuda", local_rank)
    ######################### 模型with Loss创建&权重加载&DDP模式 #################################
    print("Create model ...")
    model = build_model_loss(cfg).to(device)
    num_param = get_param_num(model)
    model_message = "Number of STD with SNCE Model Parameters:"+ str(num_param)
    # print("Number of STD with SNCE Model Parameters:", num_param, file=log, flush=True)
    log.write(model_message + "\n")
    log.flush()
    print(model_message)

    if args.weight_path != None:
        model.std.load_state_dict(torch.load(args.weight_path)['model'])
        model.criterion.load_state_dict(torch.load(args.weight_path)['criterion'])

    model = DDP(model, device_ids=[local_rank], output_device=local_rank,broadcast_buffers=False,
                                             find_unused_parameters=True)

    ######################### 数据集读取 & DDP & 加载 #################################
    train_type = cfg.DATA.TYPE     
    train_dataset = MOTTrainDataset(cfg, type= train_type)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sampler=train_sampler,
        collate_fn=collate_fn_dets,
        drop_last=False,
        shuffle=False,
    )

    ######################### 优化器 & 调节器 #################################
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(cfg, optimizer, len(train_loader))
    """
    scheduler (dict):{
            'scheduler': lr_scheduler,
            'interval': 'step',  # or 'epoch'
        }
    """
        
    ############################### 开始训练 ##################################
    if dist.get_rank() == 0:
        writer_path = os.path.join('runs', args.config_name)
        setDir(writer_path)
        writer = SummaryWriter(writer_path)

        # model_writer = SummaryWriter(cfg.MISC.MODEL_WRITER)
    # 间隔tensorbord可视化和保存模型参数的epoch
    if cfg.TRAIN.EPOCH_LOG_MOD=='interval':
        log_epochs = [i for i in range(0,cfg.TRAIN.EPOCH, cfg.TRAIN.EPOCH_LOG_INTERVAL)]
    elif cfg.TRAIN.EPOCH_LOG_MOD=='specified':
        log_epochs = cfg.TRAIN.EPOCH_LOG_SPECIFIED

    start_time = time.time()
    print("Strat training ...")

    if dist.get_rank() == 0:
        outer_bar = tqdm(total=cfg.TRAIN.EPOCH, desc="Training", position=0, colour='green')
    # -------------------epoch---------------------------
    for epoch in range(1, cfg.TRAIN.EPOCH + 1):
        if dist.get_rank() == 0:
            outer_bar.update(1)

        train_loader.sampler.set_epoch(epoch) # DDP

        start_epoch_time = time.time() # 每个epoch记录开始时间
        epoch_loss = 0. # 记录当前epoch loss
        epoch_task_loss = 0.
        epoch_snce_loss = 0.0

        model.double().train()
        # ====================== iter 循环======================
        if dist.get_rank() == 0:
            inner_bar = tqdm(total=len(train_loader), desc="Epoch {}".format(epoch), position=1)
        for i, (inps, tgts) in enumerate(train_loader):
            if dist.get_rank() == 0:
                inner_bar.update(1)

            total_loss, loss_task, loss_social_nce, outputs = model(inps, tgts)
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=cfg.TRAIN.GRADIENT_CLIP) # 梯度截断
            optimizer.step()

            if cfg.SCHEDULER.WARMUP:
                scheduler['scheduler'].iter_update()
            else:
                if scheduler['interval'] == 'step':
                    scheduler['scheduler'].step()

            epoch_loss += total_loss.item()
            epoch_task_loss += loss_task.item()
            epoch_snce_loss += loss_social_nce.item()

            # 间隔记录迭代情况,只在txt记录 减少GPU占用
            if (i + 1) % cfg.TRAIN.ITER_LOG_INTERVAL == 0:
                
                # print(
                #     "Epoch [{}/{}], Step [{}/{}]    Total Loss: {:.4f}  |  Task Loss: {:.4f}  |  Snce Loss: {:.4f}".format(
                #         epoch,
                #         cfg.TRAIN.EPOCH,
                #         i + 1,
                #         len(train_loader),
                #         total_loss.item(),
                #         loss_task.item(),
                #         loss_social_nce.item()
                #     ), file=log, flush=True
                # )
                if dist.get_rank() == 0:
                    train_iter_message = "Epoch [{}/{}], Step [{}/{}]    Total Loss: {:.4f}  |  Task Loss: {:.4f}  |  Snce Loss: {:.4f}".format(
                            epoch,
                            cfg.TRAIN.EPOCH,
                            i + 1,
                            len(train_loader),
                            total_loss.item(),
                            loss_task.item(),
                            loss_social_nce.item())
                    log.write(train_iter_message+"\n")
                    log.flush()
                    outer_bar.write(train_iter_message)

                    # for name, param in model.named_parameters():  # 返回网络的梯度和权重
                    #     model_writer.add_histogram(name + '_grad', param.grad, (epoch-1) * len(train_loader) + i)
                    #     model_writer.add_histogram(name + '_data', param, (epoch-1) * len(train_loader) + i)
                
                
            
            ######################## epoch记录#########################
            # 间隔或指定抽取某几个epoch最后一次迭代的情况进行记录 + 保存模型 + val
            if (i+1)==len(train_loader) and epoch in log_epochs:
                if dist.get_rank() == 0:
                    epoch_log(cfg, args, model, tgts, outputs, writer, epoch, 'train') #log
                    model_out_path = os.path.join(output_dir, "model_epoch_{}.pth.tar".format(epoch)) # 保存模型
                    torch.save({
                                'model': model.module.std.state_dict(),
                                'criterion':  model.module.criterion.state_dict(),
                                'optimizer': optimizer.state_dict()
                                }, model_out_path)
                    print(" Checkpoint saved to {}.".format(model_out_path))

        if dist.get_rank() == 0:
            inner_bar.close()
            # ====================== iter 循环======================

        # -------------------- epoch 记录 --------------------------
        # 每个epoch 都记录总损失和时间, loss图可视化，lr图可视化
        if cfg.SCHEDULER.WARMUP:
            if scheduler['scheduler_type'] == 'ReduceLROnPlateau':
                scheduler['scheduler'].epoch_update(epoch_loss/len(train_loader))
            else:
                scheduler['scheduler'].epoch_update()
        else:
            if scheduler['interval'] == 'epoch':
                if scheduler['scheduler_type'] == 'ReduceLROnPlateau':
                    scheduler['scheduler'].step(epoch_loss/len(train_loader))
                else:
                    scheduler['scheduler'].step()

        epoch_time = time.time() - start_epoch_time
        # print(
        #     "Epoch [{}/{}] done.    Total Loss: {:.4f}  |  Task Loss: {:.4f}  |  Snce Loss: {:.4f}.    Train Time: {}".format(
        #         epoch,
        #         cfg.TRAIN.EPOCH,
        #         epoch_loss/len(train_loader),
        #         epoch_task_loss/len(train_loader),
        #         epoch_snce_loss/len(train_loader),  
        #         epoch_time
        #     ), file=log
        # )
        if dist.get_rank() == 0:
            epoch_message = "Epoch [{}/{}] done.    Total Loss: {:.4f}  |  Task Loss: {:.4f}  |  Snce Loss: {:.4f}.    Train Time: {}".format(
                    epoch,
                    cfg.TRAIN.EPOCH,
                    epoch_loss/len(train_loader),
                    epoch_task_loss/len(train_loader),
                    epoch_snce_loss/len(train_loader),  
                    epoch_time
                )
            log.write(epoch_message+"\n")
            log.flush()
            outer_bar.write(epoch_message)
            # epoch的loss、lr图
            writer.add_scalar('Training Total Loss / Epoch',
                            epoch_loss/len(train_loader),
                            epoch)
            writer.add_scalar('Training Task Loss / Epoch',
                            epoch_task_loss/len(train_loader),
                            epoch)
            writer.add_scalar('Training Snce Loss / Epoch',
                            epoch_snce_loss/len(train_loader),
                            epoch)
            writer.add_scalar('Learning Rate / Epoch',
                            optimizer.state_dict()['param_groups'][0]['lr'],
                            epoch)
            
    torch.cuda.synchronize()    
    epoch_time = time.time() - start_time
    # print(f"{cfg.TRAIN.EPOCH}个epoch的训练总时间是：{epoch_time}", file=log, flush=True)
    finish_message = "{}个epoch的训练总时间是: {}".format(cfg.TRAIN.EPOCH,epoch_time)
    log.write(finish_message)
    log.flush()
    print(finish_message)
    
    log.close()
    if dist.get_rank() == 0:
        writer.close()
        outer_bar.close()
        # model_writer.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('Model training & evaluation')
    # ####################### exp config ########################################
    parser.add_argument('--config_name', default='default', type=str, help='Config name')
    parser.add_argument('--exp_dir', default='./experiments/', help='path where to save experiment config yaml')
    parser.add_argument('--ckpt_dir', default='./ckpts/', help='path where to save checkpoint')
    parser.add_argument('--weight_path', default=None, type=str, help='path to the .pth weight file')
    
    # ############################### system ###################################
    parser.add_argument('--gpus', default='0',help='-1 for CPU, use comma for multiple gpus')
    parser.add_argument("--local_rank", default=-1)

    # ############################### tensorboard ###############################
    parser.add_argument('--watch_model_parameters', action='store_true',
                        help='watch the parameters of model using tensorboard')
    parser.add_argument('--watch_model_gradients', action='store_true',
                        help='watch the gradients of model using tensorboard')
    parser.add_argument('--watch_contrastive_embedding', action='store_true',
                        help='watch the contrastive embedding of model using tensorboard')
    parser.add_argument('--watch_similarity_matrix', action='store_true',
                        help='watch the similarity_matrix of model using tensorboard')
    parser.add_argument('--watch_model_structure', action='store_true',
                        help='watch the structure of model using tensorboard')
    
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


    main(args)