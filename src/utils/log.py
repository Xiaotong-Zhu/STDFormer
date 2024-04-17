import torch

def epoch_log(cfg, args, model, targets, outputs, writer, epoch, type):
    # outputs: conf_matrix, tp, track_enc_emb, current_track_emb, temp_token_mask, current_track_mask
    # tensorboard 记录
    # ------------- tensorboard 记录 ----------------------------------
    if type=="train":
        if args.watch_model_parameters: # 参数记录
            for name, param in model.named_parameters():
                assert not torch.any(torch.isnan(param) + torch.isinf(param))
                writer.add_histogram(name + '_data', param, epoch)

        if args.watch_model_gradients: # 权重记录
            for name, param in model.named_parameters():  # 返回网络的梯度和权重
                assert not torch.any(torch.isnan(param.grad) + torch.isinf(param.grad))
                writer.add_histogram(name + '_grad', param.grad, epoch)

    if args.watch_contrastive_embedding: # 对比嵌入向量可视化 TODO:待实现
        pass

    if args.watch_similarity_matrix:
        assert cfg.MISC.SIMILARITY_VIS_SAMPLE<=outputs[0].shape[0]
        sample_indexes = torch.randint(low=0, high=outputs[0].shape[0],size = (cfg.MISC.SIMILARITY_VIS_SAMPLE,))
        for i in sample_indexes:
            sim_matrix = outputs[0][i]
            gt = targets['id_cls'][i]    
            writer.add_image('{}_sim_matrix_{}'.format(type,i), sim_matrix, 0, dataformats='HW')
            writer.add_image('{}_gt_{}'.format(type,i), gt, 0, dataformats='HW')



