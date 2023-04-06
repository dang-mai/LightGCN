import torch
from torch import nn
import display
from dataloaders import DataLoader
from configuration import Config


class LightGCN(nn.Module):
    def __init__(self, config: Config, dataloader: DataLoader):
        super(LightGCN, self).__init__()
        # self.config = config
        # self.dataloader = dataloader
        self.n_users = dataloader.n_users
        self.m_items = dataloader.m_items
        self.embedding_size = config.embedding_size
        self.layers = config.layers
        self.keep_prob = config.keep_prob
        self.a_hat_split = config.a_hat_split
        self.pretrain = config.pretrain
        self.dropout = config.dropout
        self.norm_adj = dataloader.norm_adj
        self.__init_weight()

    def __init_weight(self):
        # 初始嵌入
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.embedding_size)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.m_items, embedding_dim=self.embedding_size)
        # nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        # nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        # print('use xavier initilizer')
        # random normal init seems to be a better choice
        # when lightGCN actually don't use any non-linear activation function
        if self.pretrain == 0:
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            display.color_print('use NORMAL distribution initilizer')
        else:
            # self.embedding_user.weight.data.copy_(torch.from_numpy(self.weights.embedding_user))
            # self.embedding_item.weight.data.copy_(torch.from_numpy(self.weights.embedding_item))
            print('use pretarined data')
        self.sigmoid = nn.Sigmoid()
        print(f"lgn is already to go(dropout:{self.dropout})")

    @staticmethod
    def __dropout_adj(x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        # g = torch.sparse.FloatTensor(index.t(), values, size)
        g = torch.sparse_coo_tensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.a_hat_split:
            norm_adj = []
            for adj in self.norm_adj:
                norm_adj.append(self.__dropout_adj(adj, keep_prob))
        else:
            norm_adj = self.__dropout_adj(self.norm_adj, keep_prob)
        return norm_adj
    
    def computer(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.norm_adj
        else:
            g_droped = self.norm_adj
        
        for _ in range(self.layers):
            if self.a_hat_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_users, self.m_items])
        return users, items
    
    def get_users_rating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.sigmoid(torch.matmul(users_emb, items_emb.t()))
        return rating
    
    def get_embeddings(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        (users_emb, pos_emb, neg_emb, 
        userEmb0,  posEmb0, negEmb0) = self.get_embeddings(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + posEmb0.norm(2).pow(2) + negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        
        return loss, reg_loss
       
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
