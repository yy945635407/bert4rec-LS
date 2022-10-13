import torch

from options import args
from models import model_factory
from dataloaders import dataloader_factory
from trainers import trainer_factory
from utils import *
import requests

def send2vx(content, token="14bebffbc348", title="bert4rec", name="bert4rec_chscore"):
    # used for sending message to wechat server
    # see https://www.autodl.com/docs/msg/
    resp = requests.post("https://www.autodl.com/api/v1/wechat/message/push",
                        json={
                            "token": token,
                            "title": title,
                            "name": name,
                            "content": content
                        })
    print(resp.content.decode())
    return

def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader, test_loader, export_root)
    trainer.train()

    if args.template.startswith('train_bertls_clustered'):
        # push tests
        ch_score = trainer.get_train_ch()
        send2vx("n_cluster: {} n_epochs: {} chscore: {}".format(args.n_clusters, args.num_epochs, ch_score))
    test_model = True #(input('Test model with test dataset? y/[n]: ') == 'y')
    if test_model:
        trainer.test()

if __name__ == '__main__':
    if args.mode == 'train':
        train()
    else:
        raise ValueError('Invalid mode')
