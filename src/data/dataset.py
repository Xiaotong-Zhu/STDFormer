import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
from src.utils.matcher import HungarianMatcher

import glob


class Node: #'''单轨迹单节点'''
    def __init__(self, box, frame_id, next_fram_id=-1, cls=None):
        self.box = box # ndarray
        self.frame_id = frame_id
        self.next_frame_id = next_fram_id
        if cls!=None:
            self.cls = cls


class Track: #'''单轨迹'''
    def __init__(self, id):
        self.nodes = list()
        self.id = id

    def add_node(self, n):
        if len(self.nodes) > 0:
            self.nodes[-1].next_frame_id = n.frame_id
        self.nodes.append(n)

    def get_node_by_index(self, index):
        return self.nodes[index]


class Tracks: # """多轨迹"""
    def __init__(self):
        self.tracks = list()

    def add_node(self, node, id):
        node_add = False
        track_index = 0
        node_index = 0
        for t in self.tracks:
            if t.id == id:
                t.add_node(node)
                node_add = True
                track_index = self.tracks.index(t)
                node_index = t.nodes.index(node)
                break
        if not node_add:
            t = Track(id)
            t.add_node(node)
            self.tracks.append(t)
            track_index = self.tracks.index(t)
            node_index = t.nodes.index(node)

        return track_index, node_index

    def get_track_by_index(self, index):
        return self.tracks[index]


class VideoSingleParser:
    """单视频分析

    Args:
        folder ([type]): 单个视频目录
        data_type: train/train_half/val_half
        frame_nums (int, optional): 历史帧的保存数目. 默认10帧(TODO)，包括上一帧
        frame_gap (int, optional): 历史帧的帧间间隔. 默认1帧(TODO)
    """ 
    def __init__(self, folder, data_type, frame_nums=10, frame_gap=1, cost_bbox: float = 5, cost_giou: float = 2, min_giou: float = 0.5):  
        self.folder = folder # 把该视频目录路径保存为实例变量，便于后面读取帧 # './datasets/visdrone/det/trainval/uavXXX'
        self.video_name = folder.split('/')[-1] # 'uavXXX'
        self.type = data_type # 'trainval'
        self.node_nums = frame_nums - 1 # 10个历史帧信息，刨掉上一帧的信息 # 15
        self.node_gap = frame_gap # 控制抽帧间隔 # 1
        
        self.matcher = HungarianMatcher(cost_bbox, cost_giou, min_giou)  # 用于确定gt关联矩阵

        # 1. 获取图像高度、宽度,用于处理输入检测等数字归一化
        # seq_info = open(os.path.join(folder, '../../../train/{}/seqinfo.ini'.format(self.video_name))).read()
        self.seq_width = -1#int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        self.seq_height = -1#int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])
        
        # 2. 读取gt，筛选行人跟踪信息并按帧分组（gt有用信息筛选，初步处理
        if self.type=='trainval': # './datasets/visdrone/det/trainval/uavXXX'
            gt_file_path = os.path.join(folder, '../../../trainval/annotations/{}.txt'.format(self.video_name))
        elif self.type=='test': # train_half/val_half
            gt_file_path = os.path.join(folder, '../../../test/annotations/{}.txt'.format(self.video_name))
        gt_file = pd.read_csv(gt_file_path, header=None)
        # <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        gt_file = gt_file[gt_file[6] == 1] # 筛选出考虑的轨迹,是否进入考虑范围内的标志，0表示忽略，1表示active
        gt_file = gt_file[(gt_file[7] == 1) | (gt_file[7] == 4) | (gt_file[7] == 5) | (gt_file[7] == 6) | (gt_file[7] == 9)] # 筛选出其类别 pedestrain,car,van,bus and truck
        gt_group = gt_file.groupby(0) # 按帧分组
        gt_group_keys = gt_group.indices.keys()
        # self.max_frame_index = max(gt_group_keys)
        self.start_frame_index = min(gt_group_keys)
        self.seq_length = len(gt_group_keys)

        # 3. 构建一个Tracks类管理单个视频的多个轨迹，和一个recorder按帧管理每帧的跟踪信息（该视频所有轨迹信息管理）
        self.tracks = Tracks()
        self.recorder = {}
        for key in gt_group_keys:
            det = gt_group.get_group(key).values
            ids = np.array(det[:, 1]).astype(int)
            
            # <frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            # ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), 
            # van (5), truck (6), tricycle (7), awning-tricycle (8), bus (9), motor (10), others (11))   
            cls = np.array(det[:, 7]).astype(int)
            cls[cls==4] = 2
            cls[cls==5] = 3
            cls[cls==6] = 4
            cls[cls==9] = 5

            det = np.array(det[:, 2:6]).astype(float)
            det[:, :2] = det[:, :2] + det[:, 2:4]/2 # 将gt跟踪框转为(cx,cy,w,h)
            
            self.recorder[key] = list() # key为frame_id

            for id, c, d in zip(ids, cls, det):
                node = Node(d, key, cls=c)
                track_index, node_index = self.tracks.add_node(node, id)
                self.recorder[key].append((track_index, node_index))


    def _getdet(self, frame_index):
        '''
        读取当前帧的检测框
        '''
        det_path = os.path.join(self.folder, '{0:07}.txt'.format(frame_index))
        if os.path.getsize(det_path)==0:
            return np.array([])
        det = pd.read_csv(det_path, header=None).values
        # (x1, y1, w, h, obj_conf, class_conf, class_pred)
        det[:, :2] += det[:, 2:4]/2 # 将det检测框转为(cx,cy,w,h)
        det[:, 5] *= det[:, 4] # obj_conf*cls_conf
        det[:, 4] = det[:, 6]+1 # cls_id

        # (cx, cy, w, h, class_pred, obj_conf*class_conf)
        detection = det[:, :6]
        remain_inds = detection[:,5] > 0.6
        detection = detection[remain_inds,:5] # (cx, cy, w, h, class_pred)

        # TODO: 注释排序 
        # sorted_index = np.lexsort((detection[:,4],detection[:,5])) # 将det按cx，cy，w, h从小到大排列
        # detection = detection[sorted_index]
        return detection # ndarray

    def get_item(self, frame_index):
        '''
        【制作网络输入与标签label，dataset核心代码】
        输入：当前帧的检测(归一化)、上一帧所有轨迹的历史帧信息(归一化)
        输出（label）：分类分支——（N，m+1），当前帧N个检测按顺序与上一帧m个track的匹配，未匹配上的新目标为+1，但本函数只返回匹配序号对，在后续进一步处理
                       位移变换回归分支：(cx,cy,w,h),这是上一帧所有轨迹在当前帧的预测(归一化)
        '''
        frame_index += self.start_frame_index # 777- 376=401

        # 准备工作，特殊情况处理
        # if not frame_index in self.recorder: # 当前帧无跟踪框，自动递进到下一帧
        #     frame_index += 1
        #     # return None, None
        
        # if not (frame_index-1) in self.recorder: # 第一帧或符合上一个特殊情况的
        #     frame_index += 1 # 6.6,fix batch none的问题
        #     # return None, None

        while (( not frame_index in self.recorder) or (not (frame_index-1) in self.recorder)):
            frame_index += 1


        # 1. 获取当前帧检测：
        current_det = self._getdet(frame_index) # ((cx, cy, w, h, class_pred)
        while len(current_det)==0:
            frame_index += 1
            current_det = self._getdet(frame_index) # (cx, cy, w, h, class_pred)

        # 2. 获取上一帧的ids(实际track_indexes)和boxes,制作history输入和label
        
        #####################################################################
        #  容器准备工作
        ##################################################################### 
        # history 历史信息输入
        last = self.recorder[frame_index-1] # last:(ids,boxes)
        # last_ori_boxes = list() # 原始跟踪框信息，未排序
        history = list() # 存储history信息，记录所有轨迹的历史帧情况
        
        # id 分类分支label
        common_track_boxes = list() # 共同轨迹的当前帧boxes，参与iou匈牙利匹配的box

        # 位移分支label
        label_box_reg = np.zeros((len(last), 4)) # (m,4)

        #####################################################################
        #  history、label制作
        #####################################################################
        for i, (track_index, node_index) in enumerate(last): # 遍历上一帧轨迹，先将last tracks按cx，cy，w, h从小到大排列
            t = self.tracks.get_track_by_index(track_index)
            n = t.get_node_by_index(node_index)
        # TODO: 注释排序
        #     last_ori_boxes.append(n.box)
        # last_ori_boxes = np.array(last_ori_boxes)
        # sorted_index = np.lexsort((last_ori_boxes[:,3],last_ori_boxes[:,2],last_ori_boxes[:,1],last_ori_boxes[:,0])) # 将last tracks按cx，cy，w, h从小到大排列
        
        # # 排完序后，遍历每一个上一帧轨迹制作目标
        # for i,index in enumerate(sorted_index):
        #     track_index, node_index = last[index]
        #     t = self.tracks.get_track_by_index(track_index)
        #     n = t.get_node_by_index(node_index)

            #####################################################################
            #  history制作
            #####################################################################
            # 2.1 获取上一帧轨迹的历史帧信息
            history_single = list() # 保存单个轨迹的历史帧情况(包括上一帧的节点跟踪信息box+id)            
            history_single.append(list(n.box)+[n.cls]) # 上一帧的节点信息加上

            temp_node_index = node_index # 遍历历史节点，用于记录当前已遍历的最早点
            for j in range(self.node_nums): # 节点个数
                history_node_index = temp_node_index-self.node_gap
                if history_node_index < 0 : # 节点不够，重复填补最后一次节点位置信息，模拟在此处久站
                    n_idx_box = t.get_node_by_index(temp_node_index).box
                    history_single = history_single + [list(n_idx_box)+[n.cls]]*(self.node_nums-j) 
                    break
                else:
                    h_box = t.get_node_by_index(history_node_index).box
                    history_single.append(list(h_box)+[n.cls])
                temp_node_index = history_node_index
                
            history.append(history_single)

            #####################################################################
            #  轨迹预测 label制作
            #####################################################################
            # 2.2 轨迹预测回归分支
            if n.next_frame_id == frame_index: # 共同轨迹处理
                common_current_obj = t.get_node_by_index(node_index+1) # ndarray
                common_current_box = list(common_current_obj.box) + [common_current_obj.cls] # ndarray
                common_track_boxes.append(common_current_box)

                # 2.2.1 轨迹当前帧位置预测分支
                label_box_reg[i] = common_current_box[:4] # 未归一化
                # label_box_reg[i] = common_current_box-n.box # weiyi

                """
                TODO:初步默认，轨迹连续帧存在，上一帧轨迹的下一节点如果不是当前帧，就是轨迹的最后一帧
                """
            else: # 轨迹消失处理
                # 保持上一步态的位移，若是第一帧保持原位置
                if node_index==0: # 假设训练初始的预测速度位移为0
                    label_box_reg[i] = n.box
                    # label_box_reg[i] = 0.0

                else:# 位移分支:保持上一步态的位移
                    last2_node_box = t.get_node_by_index(node_index-1).box
                    label_box_reg[i, 0] = (n.box[0] - last2_node_box[0]) + n.box[0]
                    label_box_reg[i, 1] = (n.box[1] - last2_node_box[1]) + n.box[1]
                    label_box_reg[i, 2] = max( (n.box[2] - last2_node_box[2]) + n.box[2], 0.0)
                    label_box_reg[i, 3] = max( (n.box[3] - last2_node_box[3]) + n.box[3], 0.0)

                    # last2_node_box = t.get_node_by_index(node_index-1).box
                    # label_box_reg[i, 0] = (n.box[0] - last2_node_box[0])
                    # label_box_reg[i, 1] = (n.box[1] - last2_node_box[1])
                    # label_box_reg[i, 2] = max( (n.box[2] - last2_node_box[2]), 0.0)
                    # label_box_reg[i, 3] = max( (n.box[3] - last2_node_box[3]), 0.0)
            
        
        # ---------------------------------------------------
        # 4. return all values
        # 4.1 整理格式和信息，以及归一化
        # label_box_reg[:,0] /= self.seq_width # ndarray
        # label_box_reg[:,1] /= self.seq_height
        # label_box_reg[:,2] /= self.seq_width
        # label_box_reg[:,3] /= self.seq_height

        history = np.array(history)[:,:,:5] # (m, f_nums, 5)
        # history[:,:,0] /= self.seq_width
        # history[:,:,1] /= self.seq_height
        # history[:,:,2] /= self.seq_width
        # history[:,:,3] /= self.seq_height
        history_pad_mask = np.zeros((history.shape[0], history.shape[1]))
        
        # current_det[:,0] /= self.seq_width # ndarray
        # current_det[:,1] /= self.seq_height
        # current_det[:,2] /= self.seq_width
        # current_det[:,3] /= self.seq_height 

        common_track_boxes = np.array(common_track_boxes)
        # common_track_boxes[:,0] /= self.seq_width # ndarray
        # common_track_boxes[:,1] /= self.seq_height
        # common_track_boxes[:,2] /= self.seq_width
        # common_track_boxes[:,3] /= self.seq_height
        # --------------------------------------------------- 
        
        #####################################################################
        #  id分类 label制作  # 2.3
        #####################################################################
        # 2.3. id分类分支
        indices,_ = self.matcher(torch.from_numpy(current_det), torch.from_numpy(common_track_boxes)) # cost_bbox: float = 5, cost_giou: float = 2, min_giou: float = 0.5
        assign_indexes = list() # 匹配对indexes存储
        for r,c in indices:
            assign_indexes.append((r,c))

        #####################################################################
        #  返回值
        #####################################################################
        input_ = {'det':current_det, 'temp':history, 'temp_pad_mask': history_pad_mask}
        target_ = {'box_reg':label_box_reg, 'assign_indexes': assign_indexes, 'img_size': (self.seq_width, self.seq_height)}  
        # target_ = {'box_reg':label_box_reg, 'assign_indexes': assign_indexes}   
         
        
        return input_, target_

    def __len__(self):
        return self.seq_length

class VideoParser:
    def __init__(self, mot_root , # './datasets/visdrone/det'
                 detector , # 'uav'
                 type , # 'trainval'
                 frame_nums=10,  # 15
                 frame_gap=1, # 1
                 cost_bbox: float = 5,
                 cost_giou: float = 2,
                 min_giou: float = 0.5):
        """
        多视频分析
        Args:
            mot_root ([type], optional): 某个数据集的路径，如： ...//MOT17/det . Defaults to config['mot_root'].
            detector ([type], optional): 选用哪个检测器的数据集. Defaults to config['detector'].
            type ([type], optional): train/train_half/val_half.
        """                 
        
        # 1. 获取制定数据集和检测器下所有视频
        mot_root = os.path.join(mot_root, type) # './datasets/visdrone/det/trainval'
        all_folders = sorted(
            [os.path.join(mot_root, i) for i in os.listdir(mot_root)
             if os.path.isdir(os.path.join(mot_root, i))
             and i.find(detector) != -1]) # ['./datasets/visdrone/det/trainval/uavXXX','',...]

        # 2. 为每个视频创建单视频解析器
        self.parsers = [VideoSingleParser(folder, type, frame_nums, frame_gap, cost_bbox, cost_giou, min_giou) for folder in all_folders]

        # 3. get some basic information
        self.lens = [len(p) for p in self.parsers] # 每个视频的帧数list
        self.len = sum(self.lens) # 多个视频帧数和

    def __len__(self):
        # get the length of all the matching frame
        return self.len

    def __getitem__(self, item):
        if item < 0:
            return None, None
        # 1. find the parser
        total_len = 0
        index = 0 # parser的序号
        current_item = item
        for l in self.lens:
            total_len += l
            if item < total_len:
                break
            else:
                index += 1
                current_item -= l

        # 2. get items
        if index >= len(self.parsers):
            return None, None
        return self.parsers[index].get_item(current_item)

class MOTTrainDataset(data.Dataset):
    '''
    The class is the dataset for train, which read gt.txt file and rearrange them as the tracks set.
    it can be selected from the specified frame
    '''
    def __init__(self,cfg, type='train'):
        # 1. init all the variables
        self.mot_root = cfg.DATA.MOT_ROOT # "./datasets/visdrone/det"
        self.type = type # 'trainval'
        self.detector = cfg.DATA.DETECTOR # 'uav'
        self.frame_nums = cfg.DATA.HISTORY_FRAME_NUM # 15
        self.frame_gap = cfg.DATA.HISTORY_FRAME_GAP # 1
        self.max_object = cfg.DATA.MAX_OBJECT # 200
        self.cost_bbox = cfg.DATA.MATCHER.COST_BBOX # 5.0
        self.cost_giou = cfg.DATA.MATCHER.COST_GIOU # 2.0
        self.min_giou = cfg.DATA.MATCHER.MIN_GIOU # 0.5

        # 2. init GTParser
        self.parser = VideoParser(self.mot_root, self.detector, self.type, self.frame_nums, self.frame_gap, self.cost_bbox, self.cost_giou, self.min_giou)

    def __getitem__(self, item):
        input_, target_ = self.parser[item]

        if (input_ == None) & (target_ == None):
            return None,None

        # 根据batch并行训练需求，对数据做pad统一大小
        # det：(n,4)
        det_nums = input_['det'].shape[0]
        pad_det_nums = self.max_object - det_nums
        input_['det'] = np.pad(input_['det'], # input_[det] 的形状（n，5）转为固定的max_object x 5
                        [(0, pad_det_nums),
                         (0, 0)],
                        mode='constant',
                        constant_values=0)
        input_['det_pad_mask'] = np.array( [0]*det_nums + [1]*pad_det_nums )

        #temp:(m,f_nums,5),temp_pad_mask:(m,f_nums)
        track_nums = input_['temp'].shape[0]
        pad_track_nums = self.max_object - track_nums
        input_['temp'] = np.pad(input_['temp'], # (m,f_nums,5) -> (max_object, f_nums, 5)
                        [(0, pad_track_nums),
                         (0, 0),
                         (0, 0)],
                        mode='constant',
                        constant_values=0)

        input_['temp_pad_mask'] = np.pad(input_['temp_pad_mask'], # (m,f_nums) -> (m, 1+f_nums),增加token维度
                        [(0, 0),
                         (1, 0)],
                        mode='constant',
                        constant_values=0)

        input_['temp_pad_mask'] = np.pad(input_['temp_pad_mask'], # (m,1+f_nums) -> (max_object, 1+f_nums)
                        [(0, pad_track_nums),
                         (0, 0)],
                        mode='constant',
                        constant_values=1)

        '''
        有dustbin通道的id_cls
        # # id_cls : (n+1, m+1)     
        # id_cls_mask = np.zeros((self.max_object+1, self.max_object+1))
        # id_cls_mask[:det_nums, :track_nums] = 1 # 有效区
        # id_cls_mask[:, -1] = 1 # 有效区
        # id_cls_mask[-1, :] = 1 # 有效区
        # target_['id_cls_mask'] = id_cls_mask

        # id_cls = np.zeros((self.max_object+1, self.max_object+1))
        # id_cls[:, -1] = 1
        # id_cls[-1, :] = 1
        # for r,c in target_['assign_indexes']:
        #     id_cls[r, c] = 1
        #     id_cls[r,-1] = 0
        #     id_cls[-1,c] = 0
        # target_['id_cls'] = id_cls
        '''

        # id_cls : (n, m)     
        id_cls_mask = np.zeros((self.max_object, self.max_object))
        id_cls_mask[:det_nums, :track_nums] = 1 # 有效区
        target_['id_cls_mask'] = id_cls_mask



        id_cls = np.zeros((self.max_object, self.max_object))
        for r,c in target_['assign_indexes']:
            id_cls[r, c] = 1
        target_['id_cls'] = id_cls
            
        # id_cls_mask = np.zeros((self.max_object,))
        # id_cls_mask[:det_nums] = 1 # 有效区
        # target_['id_cls_mask'] = id_cls_mask

        # id_cls = np.full((self.max_object), -1)
        # for r,c in target_['assign_indexes']:
        #     id_cls[r] = c
        # target_['id_cls'] = id_cls


                  
        # box_reg: (m,4)
        target_['box_reg'] = np.pad(target_['box_reg'], # input_[space] 的形状（m，4）转为固定的max_object x 5
                        [(0, pad_track_nums),
                         (0, 0)],
                        mode='constant',
                        constant_values=0)
        target_['box_reg_mask'] = np.array( [1]*track_nums + [0]*pad_track_nums )
        target_['contrastive_lable_track'] = np.arange(self.max_object)
        target_['contrastive_lable_det'] = np.arange(self.max_object)

        return input_, target_

    def __len__(self): # 视频个数
        return len(self.parser)


class LoadDets:  # for inference
    def __init__(self, path, seq_name, type):
        '''
        path:'./datasets/visdrone/det/uavXXX'
        type: 'test'
        seq_name: 'uavXXX'
        '''
        
        self.files = sorted(glob.glob('%s/*.*' % path))

        self.nF = len(self.files)  # number of det files
        self.count = 0
        self.start_frame = int(self.files[0].split('/')[-1].split('.')[0])

        assert self.nF > 0, 'No dets found in ' + path

        # self.seq_width = None
        # self.seq_height = None
        # if seq_name == "MOT17-12-YOLOX":
        #     seq_info = open(os.path.join(path, '../../../{}/{}/seqinfo.ini'.format(type, seq_name))).read()
        #     self.seq_width = int(seq_info[seq_info.find('imWidth=') + 8:seq_info.find('\nimHeight')])
        #     self.seq_height = int(seq_info[seq_info.find('imHeight=') + 9:seq_info.find('\nimExt')])

 

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if self.count == self.nF:
            raise StopIteration

        det_path = self.files[self.count]
        if os.path.getsize(det_path)==0:
            return np.array([])
        det = pd.read_csv(det_path, header=None).values #(bx,by,w,h)
        # (x1, y1, w, h, obj_conf, class_conf, class_pred)
        det[:, :2] += det[:, 2:4]/2 # 将det检测框转为(cx,cy,w,h)
        det[:, 5] *= det[:, 4] # obj_conf*cls_conf
        det[:, 4] = det[:, 6]+1 # cls_id

        sorted_index = np.lexsort((det[:,5],det[:,4]))
        det = det[sorted_index][:,:6]

        return det

    def __getitem__(self, idx):
        idx = idx % self.nF
        det_path = self.files[idx]
        if os.path.getsize(det_path)==0:
            return np.array([])

        # (x1, y1, w, h, obj_conf, class_conf, class_pred)
        det = pd.read_csv(det_path, header=None).values
        det[:, :2] += det[:, 2:4]/2 # 将det检测框转为(cx,cy,w,h)
        det[:, 5] *= det[:, 4] # obj_conf*cls_conf
        det[:, 4] = det[:, 6]+1 # cls_id
        


        # if self.seq_width==None and self.seq_height==None:
        #     det[:, :2] += det[:, 2:4]/2 # 将det检测框转为(cx,cy,w,h)
        # else:
        #     det[:, 2:4] += det[:, :2] # lx,ly rx,ry
        #     det[:, 0] = det[:, 0].clip(0., self.seq_width)
        #     det[:, 1] = det[:, 1].clip(0., self.seq_height)
        #     det[:, 2] = det[:, 2].clip(0., self.seq_width)
        #     det[:, 3] = det[:, 3].clip(0., self.seq_height)

        #     det[:, 2:4] = det[:, 2:4] - det[:, :2]
        #     det[:, 0] = det[:, 0] + det[:, 2]*0.5
        #     det[:, 1] = det[:, 1] + det[:, 3]*0.5

        sorted_index = np.lexsort((det[:,5],det[:,4]))
        det = det[sorted_index][:,:6]
        return det # (cx, cy, w, h, cls, score)

    def __len__(self):
        return self.nF  # number of files