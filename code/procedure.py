import numpy as np
import torch
import multiprocessing
from functools import partial

import utils
from utils import timer
from dataloaders import DataLoader
from models import LightGCN
from loss import BPRLoss
from configuration import Config
import sampler
import metrics

def BPR_train_original(epoch, dataloader, model, bpr_loss: BPRLoss, config: Config, neg_k=1):
    model.train()
    with timer(name="Sample"):
        S = sampler.UniformSample_original(dataloader.train_list, dataloader.train_size, dataloader.n_users, dataloader.m_items)
    users = torch.Tensor(S[:, 0]).long()
    pos_items = torch.Tensor(S[:, 1]).long()
    neg_items = torch.Tensor(S[:, 2]).long()
    users = users.to(config.device)
    pos_items = pos_items.to(config.device)
    neg_items = neg_items.to(config.device)
    users, pos_items, neg_items = utils.shuffle(users, pos_items, neg_items)

    total_batch = len(users) // config.train_batch_size + 1
    average_loss = 0.
    for (batch_i, (batch_users, batch_pos, batch_neg)) in enumerate(utils.minibatch(config.train_batch_size, users, pos_items, neg_items)):
        batch_loss = bpr_loss.step(batch_users, batch_pos, batch_neg)
        average_loss += batch_loss
        if config.tensorboard_writer:
            config.tensorboard_writer.add_scalar(f'BPRLoss/BPR', batch_loss, epoch * int(len(users) / config.train_batch_size) + batch_i + 1)
    aver_loss = average_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(x, topks):
    """
    X: (sorted_items, groundTrue)
    sorted_items: [test_batch_size, MaxTopK]，已经按照打分排序的item，里面的item是id，numpy格式
    groundTrue: [test_batch_size, item_num]，groundTrue中的item是id，变长的嵌套list
    计算测试阶段一个test_batch的recall, precision, ndcg
    返回的是一个test_batch的TopK的recall, precision, ndcg，没有对batch进行平均
    """
    sorted_items = x[0].numpy()
    groundTrue = x[1]
    r = metrics.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = metrics.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(metrics.NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(epoch, dataloader: DataLoader, model: LightGCN, config: Config):
    # eval mode with no dropout
    model = model.eval()
    test_dict = dataloader.test_dict

    max_K = max(config.topks)

    results = {'precision': np.zeros(len(config.topks)),
               'recall': np.zeros(len(config.topks)),
               'ndcg': np.zeros(len(config.topks))}
    with torch.no_grad():
        users = list(test_dict.keys())
        try:
            assert config.test_batch_size <= len(users) / 10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # auc_record = []
        # ratings = []
        total_batch = len(users) // config.test_batch_size + 1
        for batch_users in utils.minibatch(config.test_batch_size, users):
            allPos = dataloader.get_pos_items(batch_users)
            groundTrue = [test_dict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(config.device)

            rating = model.get_users_rating(batch_users_gpu)
            #rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1<<10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            # aucs = [ 
            #         utils.AUC(rating[i],
            #                   dataset, 
            #                   test_data) for i, test_data in enumerate(groundTrue)
            #     ]
            # auc_record.extend(aucs)
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if config.multi_core:
            cores = multiprocessing.cpu_count() // 2
            pool = multiprocessing.Pool(cores)
            # pre_results = pool.map(test_one_batch, (X, topks)) # 只能传入一个参数
            partial_work = partial(test_one_batch, topks=config.topks) # 提取x作为partial函数的输入变量
            pre_results = pool.map(partial_work, X)
            pool.close()
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, config.topks))
        # scale = float(test_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        # results['auc'] = np.mean(auc_record)
        if config.tensorboard_writer:
            config.tensorboard_writer.add_scalars(f'Test/Recall@{config.topks}',
                          {str(config.topks[i]): results['recall'][i] for i in range(len(config.topks))}, epoch)
            config.tensorboard_writer.add_scalars(f'Test/Precision@{config.topks}',
                          {str(config.topks[i]): results['precision'][i] for i in range(len(config.topks))}, epoch)
            config.tensorboard_writer.add_scalars(f'Test/NDCG@{config.topks}',
                          {str(config.topks[i]): results['ndcg'][i] for i in range(len(config.topks))}, epoch)
            
        # print(results)
        return results
