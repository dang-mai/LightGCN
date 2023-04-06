from torch import optim

from models import LightGCN
from configuration import Config

class BPRLoss:
    def __init__(self, model: LightGCN, config: Config):
        self.decay = config.decay
        self.lr = config.lr
        self.model = model

        self.opt = optim.Adam(model.parameters(), lr=self.lr)

    def step(self, users, pos, neg):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg)
        reg_loss = reg_loss * self.decay
        loss = loss + reg_loss

        # 优化模型参数
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return loss.cpu().item()  # 返回loss的值