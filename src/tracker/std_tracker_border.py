import numpy as np
from collections import deque

import torch
from scipy.optimize import linear_sum_assignment

from src.models.STD import STD
from src.data.dataset import Track
from src.utils.matcher import HungarianMatcher


def next_id():
    STDTracker.count = STDTracker.count+1
    return STDTracker.count

class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3

class Track:
    def __init__(self, frame_id, box, det_score, history_buffer_size=30,track_buffer=30, sample_freq=1, img_size = None):
        """ 单个轨迹状态动作

        Args:
            frame_id (_type_): 轨迹目标出现帧id,用来判断时候连续存在进行轨迹confirmed
            box (_type_): 轨迹目标出现box,均为未归一化的
            det_score (_type_): 轨迹目标出现box得分
            history_buffer_size (int, optional): 用来存储轨迹连续时空位置信息的容器大小. Defaults to 30.
            track_buffer (int, optional): 轨迹丢失后缓存的周期. Defaults to 30.
            sample_freq: 时序信息采样间隔
        """        
        # node :cx,cy,w,h
        self.nodes = deque([], maxlen=history_buffer_size)
        self.history_buffer_size = history_buffer_size
        self.track_buffer = track_buffer
        self.start_frame = frame_id # 该轨迹初始存在的帧号
        self.add_node(box)
        self.state = TrackState.New
        # TODO: 以检测得分为设置
        self.score = det_score
        
        self.id = -1 # 初始化后才赋值

        self.sample_freq = sample_freq

        self.lost_node = None

        self.height, self.width = img_size # 480,640

    def add_node(self, n): # 每帧匹配上了添加node,输入的是ndarray，转为list
        # n = n.tolist()
        self.nodes.append(n)

    def get_tracklet_reverse(self): # history通道输入
        tracklet = []
        for i in range(len(self.nodes)-1,-1,-self.sample_freq):
            t = list(self.nodes[i])
            t.append(self.id)
            tracklet.append(t)

        if len(tracklet) < self.history_buffer_size: # 补充初始位置
            start_node = tracklet[-1]
            tracklet = tracklet + [start_node]*(self.history_buffer_size - len(tracklet))
        
        return tracklet

    def get_result(self, frame_id): # 获取轨迹最新帧的匹配结果，统一格式输出
        x, y, w, h, c = self.nodes[-1]
        left = x - w/2
        top = y - h/2

        c = int(c)
        if c==2:
            c=4
        elif c==3:
            c=5
        elif c==4:
            c=6
        elif c==5:
            c=9
        
    
        #<frame_index>,<target_id>,<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

        result = '{frame},{id},{x},{y},{w},{h},{s},{cls},-1,-1\n'.format(frame=frame_id, id=self.id, x=left, y=top, w=w, h=h, s=self.score, cls=c)
        return result

    def __len__(self): # 轨迹当前已存在的跟踪点数     
        return len(self.nodes)

    def update_score(self, score):# 更新最新帧匹配分
        self.score = score
    
    def activate(self, frame_id): # 不确定的track都要进行激活检测，符合条件激活
        if (frame_id == 1) or ((frame_id - self.start_frame)==2): # 初始帧和连续存在三帧可激活，并初始化轨迹id
           self.state = TrackState.Tracked
           self.id = next_id()
    
    def lost(self, frame_id): # 本为tracked态的track无匹配要标记lost，记录跟丢位置
        self.end_frame = frame_id
        self.state = TrackState.Lost
        self.lost_node = self.nodes[-2]
    
    def re_activate(self): # 跟丢的lost重新匹配上要重新激活
        self.state = TrackState.Tracked

    def remove(self, frame_id, box):
        if self.state == TrackState.Lost:
            if(frame_id - self.end_frame - 1) == self.track_buffer or (box[0]<50 or box[0]>(self.width-50) or box[1]<50 or box[1]>self.height-50): # 超龄，删除
                self.state = TrackState.Removed
            # elif (frame_id - self.end_frame - 1) >= 15:
            #     self.add_node(self.lost_node)
            #     for i in range(len(self.nodes)):
            #         self.nodes[i] = self.lost_node
            else: #未超龄的，加预测node
                self.add_node(box)
        if self.state == TrackState.New:
            self.state = TrackState.Removed

class STDTracker(object):
    count = 0

    def __init__(self, cfg, device,img_size):

        self.frame_id = 0 # 跟踪器内部的帧数计算系统

        self.cfg = cfg
        self.device = device
        print('Creating model...')
        self.model = STD(self.cfg) # 加载模型

        # 加载训练权重
        state_dict = torch.load(self.cfg.MODEL.WEIGHT_PATH)['model'] # cfg.MODEL.CKPT:模型权重压缩包
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.double().eval()

        # 上一帧轨迹跟踪记录器：跟踪， 跟丢，不确定的三种
        self.tracked_stracks = []  
        self.lost_stracks = []  
        self.unconfirmed_stracks = []  

        # self.det_thresh = 0.3
        # self.conf_thr = 0.5
        self.matcher_I = HungarianMatcher(self.cfg.MATCHER.COST_BBOX, self.cfg.MATCHER.COST_GIOU, self.cfg.MATCHER.MIN_GIOU_I)
        self.matcher_II = HungarianMatcher(self.cfg.MATCHER.COST_BBOX, self.cfg.MATCHER.COST_GIOU, self.cfg.MATCHER.MIN_GIOU_II)
        self.matcher_III = HungarianMatcher(self.cfg.MATCHER.COST_BBOX, self.cfg.MATCHER.COST_GIOU, self.cfg.MATCHER.MIN_GIOU_III)
    
        self.img_size = img_size

        self.det_thresh = 0.6
        self.conf_thr = 1.0

        self.track_buffer = self.cfg.DATA.TRACK_BUFFER
         

    def update(self, outputs):
        self.frame_id += 1

        if len(outputs) > 0:
            '''Detections # (cx, cy, w, h, cls, score)'''
            scores = outputs[:,5]

            remain_inds = scores > self.det_thresh
            inds_low = scores > 0.1
            inds_high = scores < self.det_thresh
            
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = outputs[inds_second, :5]

            dets = outputs[remain_inds, :5]
            scores = scores[remain_inds]

 
            detections = [Track(self.frame_id, det, scores[i], self.cfg.DATA.HISTORY_FRAME_NUM, self.track_buffer, self.cfg.DATA.HISTORY_FRAME_GAP, self.img_size) for i,det in enumerate(dets)]
        else:
            detections = []
            self.lost_stracks = self.tracked_stracks + self.lost_stracks
            self.tracked_stracks = []
            self.unconfirmed_stracks = []
            return None


        # 第一帧的检测全部直接激活初始化id，加入tracked_track,然后跳过
        if self.frame_id == 1: 
            for det in detections:
                det.activate(self.frame_id)
    
            self.tracked_stracks = detections
            return self.tracked_stracks
        
        #########################################################################################
        # 1. 准备网络输入
        #########################################################################################
        inputs = {}

        inp_dets = dets.copy() # ndaaray 
        # inp_dets[:, 0] /= self.img_size[1]
        # inp_dets[:, 1] /= self.img_size[0]
        # inp_dets[:, 2] /= self.img_size[1]
        # inp_dets[:, 3] /= self.img_size[0]
        inp_dets = torch.tensor(inp_dets)
        inputs['det'] = inp_dets.unsqueeze(0) # (1,num_det,5)

        # --------------------------------------------------------------------------------
        joint_tracks = self.tracked_stracks + self.lost_stracks + self.unconfirmed_stracks 
        # --------------------------------------------------------------------------------
        temp = [t.get_tracklet_reverse() for t in joint_tracks] # list
        inp_temp = np.array(temp)[...,:5]
        inputs['temp'] = torch.tensor(inp_temp) # (last, 30, 5)
     
        tracked_num = len(self.tracked_stracks)
        lost_num = len(self.lost_stracks)

        #########################################################################################
        # 2.获取网络预测输出
        #########################################################################################
        with torch.no_grad():
            inps={key:inputs[key].to(self.device) for key in inputs}
            conf_matrix, tp = self.model(inps, is_train = False)
            conf_matrix = conf_matrix.squeeze(0).cpu() # D, T
            tp = tp.squeeze(0).cpu() # D, 5 # (cx,cy,w,h,cls)

        #########################################################################################
        # 3. conf矩阵初匹配
        #########################################################################################
        matches=[[],[]]
        matches_second=[[],[]]
        mconf = []
        mconf_second = []
        D, T = conf_matrix.shape
        um_dets_ids = [i for i in range(D)]
        um_tras_ids = [j for j in range(T)]
        um_dets_II_ids = [ii for ii in range(len(dets_second))]

        # mask = conf_matrix > self.conf_thr
        # mask = mask \
        #     * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0]) \
        #     * (conf_matrix == conf_matrix.max(dim=0, keepdim=True)[0])
        # mask_v, all_j_ids = mask.max(dim=1)
        # i_ids = torch.where(mask_v)[0].tolist()
        # j_ids = all_j_ids[i_ids].tolist()
        # mconf = conf_matrix[i_ids, j_ids].tolist()

        cost = (1-conf_matrix)#[:, :tracked_num + lost_num]
        row_ind, col_ind = linear_sum_assignment(cost)
        indices = [(r,c) for r,c in zip(row_ind, col_ind) if cost[r,c] <= 1-self.conf_thr]  # 定义GIOU Loss = 1 - GIOU，注意到GIOU范围在[-1, 1]，那么GIOU Loss的范围在[0, 2]
        mconf = [conf_matrix[r,c] for r,c in indices]
        matcher_indices = [[],[]]
        
        for r,c in indices:
            matcher_indices[0].append(r)
            matcher_indices[1].append(c)

        i_ids = [um_dets_ids[i] for i in matcher_indices[0]]
        j_ids = [um_tras_ids[j] for j in matcher_indices[1]]


        matches[0] += i_ids
        matches[1] += j_ids
        um_dets_ids = list(set(um_dets_ids) - set(i_ids)) # 更新未匹配的det和track
        um_tras_ids = list(set(um_tras_ids) - set(j_ids))

        #########################################################################################
        # 4. tp轨迹预测iou二次匹配
        #########################################################################################
        # 4.1 higher_um_detections与tracked+lost_track预测的位置
        #########################################################################################
        um_dets = inp_dets[um_dets_ids]
        # um_tras_ids_filter = [j for j in um_tras_ids if j < tracked_num + lost_num]
        um_tras_ids_filter = [j for j in um_tras_ids]
        um_tras = tp[um_tras_ids_filter]
        indices, mconf_tp_I = self.matcher_I(um_dets, um_tras, conf_matrix[um_dets_ids,:][:,um_tras_ids_filter])
        matcher_indices = [[],[]]
        
        for r,c in indices:
            matcher_indices[0].append(r)
            matcher_indices[1].append(c)

        i_ids = [um_dets_ids[i] for i in matcher_indices[0]]
        j_ids = [um_tras_ids[j] for j in matcher_indices[1]]
        matches[0] += i_ids
        matches[1] += j_ids
        um_dets_ids = list(set(um_dets_ids) - set(i_ids)) # 更新未匹配的det和track
        um_tras_ids = list(set(um_tras_ids) - set(j_ids))

        mconf += mconf_tp_I

        ########################################################################################
        # 4.2 lower_um_detections与um tracked预测的位置
        ########################################################################################
        um_dets = torch.tensor(dets_second)
        # um_dets[:, 0] /= self.img_size[1]
        # um_dets[:, 1] /= self.img_size[0]
        # um_dets[:, 2] /= self.img_size[1]
        # um_dets[:, 3] /= self.img_size[0]

        # um_tras_ids_filter = [j for j in um_tras_ids if j < tracked_num + lost_num]
        um_tras_ids_filter = [j for j in um_tras_ids]
        um_tras = tp[um_tras_ids_filter]

        indices, mconf_tp_II = self.matcher_II(um_dets, um_tras)
        matcher_indices = [[],[]]

        for r,c in indices:
            matcher_indices[0].append(r)
            matcher_indices[1].append(c)

        i_ids = [um_dets_II_ids[i] for i in matcher_indices[0]]
        j_ids = [um_tras_ids[j] for j in matcher_indices[1]]
        matches_second[0] = i_ids
        matches_second[1] = j_ids
        um_dets_II_ids = list(set(um_dets_II_ids) - set(i_ids)) # 更新未匹配的det和track
        um_tras_ids = list(set(um_tras_ids) - set(j_ids))

        mconf_second = mconf_tp_II

        #########################################################################################
        # 4.3 higher_um_detections与unconfirmed track预测的位置
        #########################################################################################
        um_dets = inp_dets[um_dets_ids]
        # um_tras_ids_filter = [j for j in um_tras_ids if j >= tracked_num + lost_num]
        um_tras_ids_filter = [j for j in um_tras_ids]
        um_tras = tp[um_tras_ids_filter]
        indices, mconf_tp_III = self.matcher_III(um_dets, um_tras, conf_matrix[um_dets_ids,:][:,um_tras_ids_filter])
        matcher_indices = [[],[]]
        
        for r,c in indices:
            matcher_indices[0].append(r)
            matcher_indices[1].append(c)

        i_ids = [um_dets_ids[i] for i in matcher_indices[0]]
        j_ids = [um_tras_ids[j] for j in matcher_indices[1]]
        matches[0] += i_ids
        matches[1] += j_ids
        um_dets_ids = list(set(um_dets_ids) - set(i_ids)) # 更新未匹配的det和track
        um_tras_ids = list(set(um_tras_ids) - set(j_ids))

        mconf += mconf_tp_III
        
        #########################################################################################
        # 5. 匹配处理
        #########################################################################################
        for i, (det_id, tra_id) in enumerate(zip(matches[0], matches[1])):
            joint_tracks[tra_id].add_node(dets[det_id]) # 给匹配上的加node
            joint_tracks[tra_id].update_score(mconf[i]) # 更新当前帧score

            # 1. tracked_track匹配上的保持原状
            # 2. lost_track匹配上的，要重新激活
            if (tracked_num <= tra_id) and (tra_id < (tracked_num + lost_num)):
                joint_tracks[tra_id].re_activate()
            # 3. unconfirmed_track匹配上的，需要测试下能否激活
            elif tra_id >= (tracked_num + lost_num):
                joint_tracks[tra_id].activate(self.frame_id)

        for ii, (det_II_id, tra_id) in enumerate(zip(matches_second[0], matches_second[1])):
            joint_tracks[tra_id].add_node(dets_second[det_II_id]) # 给匹配上的加node
            joint_tracks[tra_id].update_score(mconf_second[ii]) # 更新当前帧score

            # 1. tracked_track匹配上的保持原状
            # 2. lost_track匹配上的，要重新激活
            if (tracked_num <= tra_id) and (tra_id < (tracked_num + lost_num)):
                joint_tracks[tra_id].re_activate()
            # 3. unconfirmed_track匹配上的，需要测试下能否激活
            elif tra_id >= (tracked_num + lost_num):
                joint_tracks[tra_id].activate(self.frame_id)

        #########################################################################################
        # 6. 未匹配处理
        #########################################################################################
        # 6.1 未匹配的track处理
        #########################################################################################
        # tp[:, 0] *= self.img_size[1]
        # tp[:, 1] *= self.img_size[0]
        # tp[:, 2] *= self.img_size[1]
        # tp[:, 3] *= self.img_size[0]
        tp = tp.numpy()

        for tra_id in um_tras_ids:
            # 1. tracked没匹tra_ids配，变lost，加预测的box作为node
            if tra_id < tracked_num: 
                joint_tracks[tra_id].add_node(tp[tra_id])
                joint_tracks[tra_id].update_score(-1) # 更新当前帧score
                joint_tracks[tra_id].lost(self.frame_id)
            # 2. lost和unconfirmed没匹配的，用remove测试一下，lost两种处理：(1)超龄的，删除；(2)未超龄的，加预测的node，保持在lost里
            else:
                joint_tracks[tra_id].remove(self.frame_id, tp[tra_id])

        #########################################################################################
        # 7. 重新确认tracked、lost、unconfirmed
        #########################################################################################
        new_tracked_stracks = [] 
        new_lost_stracks = []  
        new_unconfirmed_stracks = []

        for i in joint_tracks:
            if i.state == TrackState.Tracked:
                new_tracked_stracks.append(i)
            elif i.state == TrackState.Lost:
                new_lost_stracks.append(i)
            elif i.state == TrackState.New:
                new_unconfirmed_stracks.append(i)

        #########################################################################################
        # 6.1 未匹配的det处理，把他加进unconfirmed_track
        #########################################################################################
        for det_id in um_dets_ids:
            new_unconfirmed_stracks.append(detections[det_id])
        
        self.tracked_stracks = new_tracked_stracks
        self.lost_stracks = new_lost_stracks
        self.unconfirmed_stracks = new_unconfirmed_stracks

        #########################################################################################
        # 8. 返回跟踪到的track结果
        #########################################################################################
        return new_tracked_stracks

    def reset_count(self):
        STDTracker.count = 0