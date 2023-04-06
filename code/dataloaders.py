import torch
import numpy as np
import scipy.sparse as sp
from time import time

from configuration import Config
import display

class DataLoader():
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """
    def __init__(self, config: Config, path="../data/gowalla"):
        # train or test
        self.device = config.device
        display.color_print(f'loading [{path}]')
        self.split = config.a_hat_split
        self.folds = config.a_hat_folds
        self.mode_dict = {'train': 0, "test": 1}
        self.mode = self.mode_dict['train']
        self.n_users = 0
        self.m_items = 0
        train_file = path + '/train.txt'
        test_file = path + '/test.txt'
        self.path = path
        trainUniqueUsers, train_items, train_users = [], [], []  # COO格式
        testUniqueUsers, test_items, test_users = [], [], []  # COO格式
        self.train_size = 0  # 交互数
        self.test_size = 0  # 交互数

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    train_users.extend([uid] * len(items))
                    train_items.extend(items)
                    self.m_items = max(self.m_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.train_size += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.train_users = np.array(train_users)
        self.train_items = np.array(train_items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    test_users.extend([uid] * len(items))
                    test_items.extend(items)
                    self.m_items = max(self.m_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.test_size += len(items)
        self.m_items += 1
        self.n_users += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.test_users = np.array(test_users)
        self.test_items = np.array(test_items)
        
        print(f"{self.train_size} interactions for training")
        print(f"{self.test_size} interactions for testing")
        print(f"{config.dataset_name} Sparsity : {(self.train_size + self.test_size) / self.n_users / self.m_items}")

        # 邻接矩阵构建
        self.interaction_matrix = sp.csr_matrix((np.ones(len(self.train_users)), (self.train_users, self.train_items)),
                                      shape=(self.n_users, self.m_items))
        self.norm_adj = self.get_norm_adj()
        # self.users_D = np.array(self.interaction_matrix.sum(axis=1)).squeeze()
        # self.users_D[self.users_D == 0.] = 1
        # self.items_D = np.array(self.interaction_matrix.sum(axis=0)).squeeze()
        # self.items_D[self.items_D == 0.] = 1.
        # pre-calculate
        self.train_list = self.get_pos_items(list(range(self.n_users)))
        self.test_dict = self.build_test()
        print(f"{config.dataset_name} is ready to go")

    def split_a_hat(self, a):
        a_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            a_fold.append(self.convert_sp_mat_to_sp_tensor(a[start:end]).coalesce().to(self.device))
        return a_fold

    def convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        # return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
        return torch.sparse_coo_tensor(index, data, torch.Size(coo.shape))
        
    def get_norm_adj(self):
        print("loading adjacency matrix")
        try:
            pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
            print("successfully loaded...")
            norm_adj = pre_adj_mat
        except :
            print("generating adjacency matrix")
            s = time()
            adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.interaction_matrix.tolil()
            adj_mat[:self.n_users, self.n_users:] = R
            adj_mat[self.n_users:, :self.n_users] = R.T
            adj_mat = adj_mat.todok()
            # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])
            
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            end = time()
            print(f"costing {end-s}s, saved norm_mat...")
            sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)

        if self.split == True:
            norm_adj = self.split_a_hat(norm_adj)
            print("done split matrix")
        else:
            norm_adj = self.convert_sp_mat_to_sp_tensor(norm_adj)
            norm_adj = norm_adj.coalesce().to(self.device)
            print("don't split the matrix")
        return norm_adj

    def build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.test_items):
            user = self.test_users[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def get_pos_items(self, users):
        posItems = []
        for user in users:
            posItems.append(self.interaction_matrix[user].nonzero()[1])
        return posItems

