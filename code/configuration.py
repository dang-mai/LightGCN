import os
import sys
import time
import torch
from tensorboardX import SummaryWriter

# # let pandas shut up
# from warnings import simplefilter
# simplefilter(action="ignore", category=FutureWarning)

class Config():
    def __init__(self, args):
        # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 是否必须？
        # 路径设置
        self.root_path = os.path.dirname(os.path.dirname(__file__))
        self.code_path = os.path.join(self.root_path, 'code')
        self.data_path = os.path.join(self.root_path, 'data')
        self.tensorboard_path = os.path.join(self.code_path, 'runs')
        self.file_path = os.path.join(self.code_path, 'checkpoints')
        sys.path.append(os.path.join(self.code_path, 'sources'))
        # if not os.path.exists(config['tensorboard_path']):
        #     os.makedirs(config['tensorboard_path'], exist_ok=True)
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path, exist_ok=True)
        
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        # self.cores = multiprocessing.cpu_count() // 2
        self.seed = args.seed
        
        # 数据集配置&&模型配置
        self.dataset_name = args.dataset
        self.model_name = args.model
        
        # 超参数配置
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.embedding_size = args.embedding_size
        self.layers = args.layers
        self.dropout = args.dropout
        self.keep_prob = args.keep_prob
        self.a_hat_folds = args.a_hat_folds
        self.multi_core = args.multi_core
        self.lr = args.lr
        self.decay = args.decay
        self.pretrain = args.pretrain
        self.a_hat_split = False
        self.bigdata = False

        # 其它配置
        self.epochs = args.epochs
        self.weights_path = args.weights_path
        self.load = args.load
        self.topks = eval(args.topks)
        self.tensorboard = args.tensorboard
        if self.tensorboard:
            self.tensorboard_writer = SummaryWriter(os.path.join(self.tensorboard_path, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + self.model_name + "-" + self.dataset_name))
        else:
            self.tensorboard_writer = None
        
    def __str__(self):
        print('===========config================')
        print("model:", self.model_name)
        print("dataset:", self.dataset_name)
        ...
        print('===========end===================')
        return "" # 必须返回一个空字符串，是否有更优美的做法
