from .base import AbstractTrainer
from .utils import recalls_and_ndcgs_for_ks
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import AverageMeterSet
import torch
from sklearn.metrics import calinski_harabasz_score
import os
import json

class BERTLSTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader, export_root):
        super().__init__(args, model, train_loader, val_loader, test_loader, export_root)
        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        self.mse = nn.MSELoss()
        self.num_epochs_pretrain = args.num_epochs_pretrain

    @classmethod
    def code(cls):
        return 'bert_ls'

    def add_extra_loggers(self):
        pass

    def log_extra_train_info(self, log_data):
        pass

    def log_extra_val_info(self, log_data):
        pass
    
    def train(self):
        accum_iter = 0
        self.validate(0, accum_iter)
        if self.num_epochs_pretrain > 0:
            print("===Start Pretraining===")
            for epoch in range(self.num_epochs_pretrain):
                accum_iter = self.train_one_epoch(epoch, accum_iter, pretrain=True)
                self.validate(epoch, accum_iter)
            print("===Init clustering===")
            self.init_cluster()
            print("===End of pretraining===")

        for epoch in range(self.num_epochs):
            accum_iter = self.train_one_epoch(epoch, accum_iter)
            self.validate(epoch, accum_iter)
        self.logger_service.complete({
            'state_dict': (self._create_state_dict()),
        })
        self.writer.close()

    def init_cluster(self):
        # Initialize clusters in self.kmeans after pre-training
        batch_X = []
        for batch_idx, batch in enumerate(self.train_loader):
            batch = [x.to(self.device) for x in batch]
            seqs, labels = batch
            # get latent_X in AutoEncoder
            _, _, latent_X, _ = self.model(seqs)
            latent_X = latent_X.reshape(-1, latent_X.shape[-1])
            batch_X.append(latent_X.detach().cpu().numpy())
        batch_X = np.vstack(batch_X)
        self.model.kmeans.init_cluster(batch_X)

    def calculate_loss(self, batch, pretrain=False):
        if pretrain:
            seqs, labels = batch
            # logits: BERT output
            # emb: BERT embedding
            # latent: AutoEncoder latent vector
            # rec_emb: output of AutoEncoder
            logits, emb, latent, rec_emb = self.model(seqs)  # B x T x V

            logits = logits.view(-1, logits.size(-1))  # (B*T) x V
            labels = labels.view(-1)  # B*T
            bert_loss = self.ce(logits, labels)
            rec_loss = self.mse(emb, rec_emb) # reconstruction loss
            
            loss = bert_loss + rec_loss
            return {'loss': loss, 'bert_loss': bert_loss, \
                    'rec_loss': rec_loss}
        else:
            seqs, labels = batch
            logits, emb, latent, rec_emb = self.model(seqs)  # B x T x V

            logits = logits.view(-1, logits.size(-1))  # (B*T) x V
            labels = labels.view(-1)  # B*T
            bert_loss = self.ce(logits, labels)
            rec_loss = self.mse(emb, rec_emb) # reconstruction loss

            # update cluster
            # Get the latent features
            with torch.no_grad():
                _, _, latent_X, _ = self.model(seqs)
                latent_X = latent_X.reshape(-1, latent_X.shape[-1])

            # [Step-1] Update the assignment results
            cluster_id = self.model.kmeans.update_assign(latent_X)

            # [Step-2] Update clusters in batch Kmeans
            elem_count = torch.bincount(cluster_id,
                                     minlength=self.args.n_clusters)
            for k in range(self.args.n_clusters):
                # avoid empty slicing
                if elem_count[k] == 0:
                    continue
                self.model.kmeans.update_cluster(latent_X[cluster_id == k], k)

            # Regularization term on clustering
            batch_size = seqs.shape[0]
            cluster_loss = torch.tensor(0.).to(self.device)
            clusters = self.model.kmeans.clusters
            for i in range(batch_size):
                diff_vec = latent[i] - clusters[cluster_id[i]]
                sample_cluster_loss = torch.matmul(diff_vec.view(1, -1),
                                                diff_vec.view(-1, 1))
                cluster_loss += torch.squeeze(sample_cluster_loss)
            loss = bert_loss + rec_loss + cluster_loss
            return {'loss': loss, 'bert_loss': bert_loss, \
                    'rec_loss': rec_loss, 'cluster_loss': cluster_loss}

    def calculate_metrics(self, batch):
        seqs, candidates, labels = batch
        scores, emb, latent, rec_emb = self.model(seqs)  # B x T x V
        scores = scores[:, -1, :]  # B x V
        scores = scores.gather(1, candidates)  # B x C

        metrics = recalls_and_ndcgs_for_ks(scores, labels, self.metric_ks)

        return metrics

    def calculate_ch(self, batch):
        seqs, _ = batch
        _, _, latent, _ = self.model(seqs)  # B x T x V
        metrics = {}
        # add ch index
        latent = latent.reshape(-1, latent.shape[-1])
        clusters = self.model.kmeans.update_assign(latent).cpu().numpy()
        latent = latent.cpu().numpy()
        if np.count_nonzero(clusters) >= 1:
            metrics['chscore'] = calinski_harabasz_score(latent, clusters)
        return metrics

    def get_train_ch(self):
        # get metrics with ch score on train set
        print('Test best model with test set!')

        best_model = torch.load(os.path.join(self.export_root, 'models', 'best_acc_model.pth')).get('model_state_dict')
        self.model.load_state_dict(best_model)
        self.model.eval()

        average_meter_set = AverageMeterSet()

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.train_loader)
            for batch_idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]

                metrics = self.calculate_ch(batch)

                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                description = ""
                if 'chscore' in metrics:
                    description += " CH score {:.3f}".format(metrics['chscore'])
                tqdm_dataloader.set_description(description)

            average_metrics = average_meter_set.averages()
            with open(os.path.join(self.export_root, 'logs', 'test_metrics.json'), 'w') as f:
                json.dump(average_metrics, f, indent=4)
            print(average_metrics)
        return average_metrics['chscore']