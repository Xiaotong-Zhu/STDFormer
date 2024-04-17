import os
import torch
from yacs.config import CfgNode as CN

from src.data.dataset import LoadDets
from src.tracker.std_tracker_border import STDTracker

import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def mkdirs(d):
    if not os.path.exists(d):
        os.makedirs(d)

def main(args):
    torch.backends.cudnn.enable =True
    torch.backends.cudnn.benchmark = True

    default_yaml = './src/config/track.yaml'
    default_cfg = open(default_yaml)
    cfg = CN.load_cfg(default_cfg)
    if args.config_name!='default':
        cfg.merge_from_file(os.path.join(args.exp_dir, args.config_name, "track.yaml"))
    if args.result_dir!=None:
        cfg.RESULT.PATH = args.result_dir
    if args.weight_path!=None:
        cfg.MODEL.WEIGHT_PATH = args.weight_path    
    cfg.freeze()

    det_root = os.path.join(cfg.DATA.MOT_ROOT, cfg.DATA.TYPE)
    all_folders = sorted([os.path.join(det_root, i) for i in os.listdir(det_root)
            if os.path.isdir(os.path.join(det_root, i))
            and i.find(cfg.DATA.DETECTOR) != -1]) # 视频文件目录路径

    for seq_path in all_folders:
        seq_name = seq_path.split('/')[-1]
        dataloader = LoadDets(seq_path, seq_name, cfg.DATA.TYPE) # (cx, cy, w, h, cls, score)

        # #debug
        # if seq_name != 'MOT17-03-YOLOX':
        #     continue

        # 读取第一张图
        img_path = os.path.join(cfg.DATA.MOT_ROOT,'..', cfg.DATA.TYPE, 'sequences', seq_name, '0000001.jpg')
        # print(img_path)
        img = cv2.imread(img_path)
        img_size = img.shape[:2]

        
        mkdirs(cfg.RESULT.PATH)
        result_filename = os.path.join(cfg.RESULT.PATH, '{}.txt'.format(seq_name))# 以视频为单位存储跟踪结果
        
        tracker = STDTracker(cfg, device, img_size)
        results = []
        frame_id = dataloader.start_frame - 1


        for dets in dataloader: # 跟踪每一帧 # (cx, cy, w, h, cls, score)
            frame_id += 1
            new_tracked_stracks = tracker.update(dets)
            if new_tracked_stracks == None:
                continue

            for i in new_tracked_stracks:
                result = i.get_result(frame_id)
                results.append(result)

        tracker.reset_count() 
        with open(result_filename, 'w') as f:
            for line in results:
                f.write(line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser('track')
    # ####################### exp config ########################################
    parser.add_argument('--config_name', default='default', type=str, help='Config name')
    parser.add_argument('--exp_dir', default='./experiments/', help='path where to save experiment config yaml')
    parser.add_argument('--result_dir', default=None, help='path where to save results')
    parser.add_argument('--weight_path', default=None, type=str, help='path to the .pth weight file')
    
    args = parser.parse_args()
    main(args)