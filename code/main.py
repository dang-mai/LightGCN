import torch
import time
import os.path

import utils
import display
import register
import dataloaders
import loss
import procedure
from parse import parse_args
from configuration import Config


logo = r"""
██╗     ██╗ ██████╗ ██╗  ██╗████████╗ ██████╗  ██████╗███╗   ██╗
██║     ██║██╔════╝ ██║  ██║╚══██╔══╝██╔════╝ ██╔════╝████╗  ██║
██║     ██║██║  ███╗███████║   ██║   ██║  ███╗██║     ██╔██╗ ██║
██║     ██║██║   ██║██╔══██║   ██║   ██║   ██║██║     ██║╚██╗██║
███████╗██║╚██████╔╝██║  ██║   ██║   ╚██████╔╝╚██████╗██║ ╚████║
╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝    ╚═════╝  ╚═════╝╚═╝  ╚═══╝
"""

if __name__ == '__main__':
    # 加载配置
    args = parse_args()
    config = Config(args)
    display.color_print(logo)
    print(config)
    display.pprint(config.__dict__)

    # 初始化设置
    # init tensorboard
    if not config.tensorboard:
        display.color_print("not enable tensorflowboard")
    # 固定随机种子
    utils.set_seed(config.seed)
    print(">>SEED:", config.seed)

    # login
    # 登录模型与数据加载器
    if config.dataset_name in register.DATASETS.keys():
        dataloader = dataloaders.DataLoader(config, path="../data/" + config.dataset_name)
        pass
    else:  # args.dataset == 'other datasets'
        raise NotImplementedError(f"Haven't supported {config.dataset_name} yet!")
    
    if config.model_name in register.MODELS.keys():
        model = register.MODELS[config.model_name](config, dataloader)
    else:  # args.model == 'other models'
        raise NotImplementedError(f"Haven't supported {config.model_name} yet!")

    model = model.to(config.device)
    bpr_loss = loss.BPRLoss(model, config)

    # 预训练
    weight_filename = f"{config.model_name}-{config.dataset_name}-{config.layers}-{config.embedding_size}.pth.tar"
    weight_filename = os.path.join(config.file_path, weight_filename)
    print(f"load and save to {weight_filename}")
    if config.load:
        try:
            model.load_state_dict(torch.load(weight_filename, map_location=torch.device('cpu')))
            display.color_print(f"loaded model weights from {weight_filename}")
        except FileNotFoundError:
            print(f"{weight_filename} not exists, start from beginning")

    # 训练过程
    try:
        # 训练前测试
        display.color_print("[TEST]EPOCH[0]}")
        results = procedure.Test(0, dataloader, model, config)
        print(results)

        for epoch in range(config.epochs):
            start = time.time()
            output_information = procedure.BPR_train_original(epoch, dataloader, model, bpr_loss, config, neg_k=1)
            print(f'EPOCH[{epoch + 1}/{config.epochs}] {output_information}')
            if (epoch + 1) % 10 == 0:
                display.color_print(f"[TEST]EPOCH[{epoch + 1}")
                results = procedure.Test(epoch + 1, dataloader, model, config)
                print(results)
            torch.save(model.state_dict(), weight_filename)
    finally:
        if config.tensorboard_writer:
            config.tensorboard_writer.close()
