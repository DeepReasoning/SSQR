import argparse
import time
from pprint import pprint
import numpy as np
import random
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import dgl
import wandb
from utils.process_data import load_data, load_data_new
from torch.nn.utils import clip_grad_norm_

from model import CompGCN_DistMult, CompGCN_ConvE, VQGCN, VQGCN_MLP
from utils import process, TrainDataset, TestDataset


class Runner(object):
    def __init__(self, params):
        self.p = params

        # wandb.init(project="VQKG", name='VQTest_notransformer',config=params)
        wandb.init(project="VQKG_1103", name='VQ4KG',config=params)

        self.prj_path = Path(__file__).parent.resolve()  # 目录
        # self.data = load_data(self.p.dataset)
        self.data = load_data_new(self.p.dataset)
        self.num_ent, self.train_data, self.valid_data, self.test_data, self.num_rels = self.data.num_nodes, \
            self.data.train, self.data.valid, self.data.test, self.data.num_rels
        self.triplets = process({'train': self.train_data, 'valid': self.valid_data, 'test': self.test_data},
                                self.num_rels)  # 训练集合
        self.device = torch.device(f'cuda:{self.p.gpu}')
        # self.device = torch.device('cpu')
        self.data_iter = self.get_data_iter()
        self.g = self.build_graph()
        self.edge_type, self.edge_norm = self.get_edge_dir_and_norm()

        self.g = self.g.to(self.device)

        self.model = self.get_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.p.lr, weight_decay=self.p.l2)
        self.optimizer_vq = torch.optim.Adam(self.model.codebook.parameters(), lr=self.p.lr, weight_decay=self.p.l2)  # vq训练器
        self.best_val_mrr, self.best_epoch, self.best_val_results = 0., 0., {}


        # ent_emd 3072 40943
        # self.ent_text_emd = torch.from_numpy(self.ent_text_emd).to(self.device)
        self.ent_text_emd = torch.tensor(self.ent_text_emd, requires_grad=False).float().to(self.device)
        # self.ent_text_emd = torch.randn((40943, 3072)).to(self.device)
        print(self.ent_text_emd.size(), self.ent_text_emd.dtype)
        pprint(vars(self.p))

    def fit(self):
        save_root = self.prj_path / 'checkpoints'
        if not save_root.exists():
            save_root.mkdir()
        save_path = save_root / (self.p.name + '.pt')

        if self.p.restore:
            self.load_model(save_path)
            print('Successfully Loaded previous model')


        # stage 1
        for epoch in range(self.p.max_epochs):
        # for epoch in range(1):
            start_time = time.time()
            # train_loss, pre_loss, cl_loss, q_loss = self.train_gcn()
            train_loss, pre_loss, x_loss, q_loss = self.train()
            val_results = self.evaluate('valid')
            if val_results['mrr'] > self.best_val_mrr:
                self.best_val_results = val_results
                self.best_val_mrr = val_results['mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
            print(
                f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
            wandb.log({"train_loss": train_loss, "pre_loss": pre_loss, "x_loss": x_loss, "q_loss": q_loss, "MRR": val_results['mrr'], "MR": val_results['mr'],
                       "hits@1": val_results['hits@1'], "hits@3": val_results['hits@3'], "hits@10": val_results['hits@10']})


        # # stage 2
        # for epoch in range(100):
        #     start_time = time.time()
        #     train_loss, pre_loss, cl_loss, q_loss = self.train_vq()
        #     val_results = self.evaluate('valid')
        #     # if val_results['mrr'] > self.best_val_mrr:
        #     #     self.best_val_results = val_results
        #     #     self.best_val_mrr = val_results['mrr']
        #     #     self.best_epoch = epoch
        #     #     self.save_model(save_path)
        #     print(
        #         f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
        #     wandb.log({"train_loss": train_loss, "pre_loss": pre_loss, "cl_loss": cl_loss, "q_loss": q_loss,
        #                "MRR": val_results['mrr'], "MR": val_results['mr'],
        #                "hits@1": val_results['hits@1'], "hits@3": val_results['hits@3'],
        #                "hits@10": val_results['hits@10']})
        #
        # for param_group in self.optimizer.param_groups:
        #     param_group['lr'] /= 10  # 学习率降低10倍
        #     print(param_group['lr'])
        # # stage 3
        # for epoch in range(100):
        #     start_time = time.time()
        #     train_loss, pre_loss, cl_loss, q_loss = self.train()
        #     val_results = self.evaluate('valid')
        #     if val_results['mrr'] > self.best_val_mrr:
        #         self.best_val_results = val_results
        #         self.best_val_mrr = val_results['mrr']
        #         self.best_epoch = epoch
        #         self.save_model(save_path)
        #     print(
        #         f"[Epoch {epoch}]: Training Loss: {train_loss:.5}, Valid MRR: {val_results['mrr']:.5}, Best Valid MRR: {self.best_val_mrr:.5}, Cost: {time.time() - start_time:.2f}s")
        #     wandb.log({"train_loss": train_loss, "pre_loss": pre_loss, "cl_loss": cl_loss, "q_loss": q_loss,
        #                "MRR": val_results['mrr'], "MR": val_results['mr'],
        #                "hits@1": val_results['hits@1'], "hits@3": val_results['hits@3'],
        #                "hits@10": val_results['hits@10']})


        # pprint(vars(self.p))
        self.load_model(save_path)
        print(f'Loading best model in {self.best_epoch} epoch, Evaluating on Test data')
        test_results = self.evaluate('test')
        print(
            f"MRR: Tail {test_results['left_mrr']:.5}, Head {test_results['right_mrr']:.5}, Avg {test_results['mrr']:.5}")
        print(f"MR: Tail {test_results['left_mr']:.5}, Head {test_results['right_mr']:.5}, Avg {test_results['mr']:.5}")
        print(f"hits@1 = {test_results['hits@1']:.5}")
        print(f"hits@3 = {test_results['hits@3']:.5}")
        print(f"hits@10 = {test_results['hits@10']:.5}")


        # 生成entity code并保存
        self.gen_ent_codes()
        self.gen_ent_codes_emds()



    def gen_ent_codes(self):
        self.model.eval()
        with torch.no_grad():
            codes = self.model.cal_allent_codes(self.g)
        codes = codes.cpu()
        print(codes)
        # with torch.no_grad():
        #     codes = self.model.cal_allent_codes(self.g)
        # codes = codes.cpu()
        # print(codes)
        dataset = self.p.dataset.split('/')[-1]
        save_name = f'./codes_new/{dataset}_{self.p.seq_len}_{self.p.num_code}.pt'  # GCN 2  _notext _nogcn
        torch.save(codes, save_name)

    def gen_ent_codes_emds(self):
        self.model.eval()
        with torch.no_grad():
            codes_emds = self.model.cal_allent_codes_emds(self.g)
        codes_emds = codes_emds.cpu()
        dataset = self.p.dataset.split('/')[-1]
        save_name = f'./codes_new/{dataset}_{self.p.seq_len}_{self.p.num_code}_emd.pt'  # GCN 2
        torch.save(codes_emds, save_name)


    def train(self):  # 所有整体训练
        self.model.train()
        losses = []
        pre_losses = []
        x_losses = []
        q_losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            triplets, labels = triplets.to(self.device), labels.to(self.device)
            subj, rel = triplets[:, 0], triplets[:, 1]
            pred, q_loss, x_loss = self.model(self.g, subj, rel, None, stage=3)  # [batch_size, num_ent]  ent_text
            pre_loss = self.model.calc_loss(pred, labels)
            # loss = pre_loss + q_loss + x_loss  # 正常损失
            loss = pre_loss + q_loss



            # loss = pre_loss + q_loss
            # loss = pre_loss + self.p.vq_weight * q_loss
            # loss = pre_loss
            self.optimizer.zero_grad()
            loss.backward()
            
            clip_grad_norm_(self.model.parameters(), max_norm=1)  # 梯度截断 5 2 1
            self.optimizer.step()
            losses.append(loss.item())
            pre_losses.append(pre_loss.item())
            # x_losses.append(x_loss.item())  # 没有x loss
            q_losses.append(q_loss.item())

        loss = np.mean(losses)
        pre_loss = np.mean(pre_losses)
        x_loss = np.mean(x_losses)
        q_loss = np.mean(q_losses)
        return loss, pre_loss, 0, q_loss


    def train_gcn(self):
        self.model.train()
        losses = []
        pre_losses = []
        cl_losses = []
        q_losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            triplets, labels = triplets.to(self.device), labels.to(self.device)
            subj, rel = triplets[:, 0], triplets[:, 1]
            pred, q_loss, cl_loss = self.model(self.g, subj, rel, stage=1)  # [batch_size, num_ent]
            pre_loss = self.model.calc_loss(pred, labels)
            loss = pre_loss
            self.optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(self.model.parameters(), max_norm=1)  # 梯度截断 5 2 1

            self.optimizer.step()
            losses.append(loss.item())
            pre_losses.append(pre_loss.item())
            cl_losses.append(cl_loss.item())
            q_losses.append(q_loss.item())
        loss = np.mean(losses)
        pre_loss = np.mean(pre_losses)
        cl_loss = np.mean(cl_losses)
        q_loss = np.mean(q_losses)
        return loss, pre_loss, cl_loss, q_loss


    def train_vq(self):
        self.model.train()
        losses = []
        pre_losses = []
        cl_losses = []
        q_losses = []
        train_iter = self.data_iter['train']
        for step, (triplets, labels) in enumerate(train_iter):
            triplets, labels = triplets.to(self.device), labels.to(self.device)
            subj, rel = triplets[:, 0], triplets[:, 1]
            pred, q_loss, cl_loss = self.model(self.g, subj, rel, stage=2)  # [batch_size, num_ent]
            pre_loss = self.model.calc_loss(pred, labels)
            loss = q_loss  # 仅仅是一个
            self.optimizer_vq.zero_grad()
            loss.backward()

            clip_grad_norm_(self.model.codebook.parameters(), max_norm=1)  # 梯度截断 5 2 1

            self.optimizer_vq.step()
            losses.append(loss.item())
            pre_losses.append(pre_loss.item())
            cl_losses.append(cl_loss.item())
            q_losses.append(q_loss.item())

        loss = np.mean(losses)
        pre_loss = np.mean(pre_losses)
        cl_loss = np.mean(cl_losses)
        q_loss = np.mean(q_losses)
        return loss, pre_loss, cl_loss, q_loss



    def evaluate(self, split):
        """
        Function to evaluate the model on validation or test set
        :param split: valid or test, set which data-set to evaluate on
        :return: results['mr']: Average of ranks_left and ranks_right
                 results['mrr']: Mean Reciprocal Rank
                 results['hits@k']: Probability of getting the correct prediction in top-k ranks based on predicted score
                 results['left_mrr'], results['left_mr'], results['right_mrr'], results['right_mr']
                 results['left_hits@k'], results['right_hits@k']
        """

        def get_combined_results(left, right):
            results = dict()
            assert left['count'] == right['count']
            count = float(left['count'])
            results['left_mr'] = round(left['mr'] / count, 5)
            results['left_mrr'] = round(left['mrr'] / count, 5)
            results['right_mr'] = round(right['mr'] / count, 5)
            results['right_mrr'] = round(right['mrr'] / count, 5)
            results['mr'] = round((left['mr'] + right['mr']) / (2 * count), 5)
            results['mrr'] = round((left['mrr'] + right['mrr']) / (2 * count), 5)
            for k in [1, 3, 10]:
                results[f'left_hits@{k}'] = round(left[f'hits@{k}'] / count, 5)
                results[f'right_hits@{k}'] = round(right[f'hits@{k}'] / count, 5)
                results[f'hits@{k}'] = round((results[f'left_hits@{k}'] + results[f'right_hits@{k}']) / 2, 5)
            return results

        self.model.eval()
        left_result = self.predict(split, 'tail')
        right_result = self.predict(split, 'head')
        res = get_combined_results(left_result, right_result)
        return res

    def predict(self, split='valid', mode='tail'):
        """
        Function to run model evaluation for a given mode
        :param split: valid or test, set which data-set to evaluate on
        :param mode: head or tail
        :return: results['mr']: Sum of ranks
                 results['mrr']: Sum of Reciprocal Rank
                 results['hits@k']: counts of getting the correct prediction in top-k ranks based on predicted score
                 results['count']: number of total predictions
        """
        with torch.no_grad():
            results = dict()
            test_iter = self.data_iter[f'{split}_{mode}']
            for step, (triplets, labels) in enumerate(test_iter):
                triplets, labels = triplets.to(self.device), labels.to(self.device)
                subj, rel, obj = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                pred, _, _ = self.model(self.g, subj, rel)
                b_range = torch.arange(pred.shape[0], device=self.device)
                target_pred = pred[b_range, obj]  # [batch_size, 1], get the predictive score of obj
                # label=>-1000000, not label=>pred, filter out other objects with same sub&rel pair
                pred = torch.where(labels.byte(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred  # copy predictive score of obj to new pred
                ranks = 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj]  # get the rank of each (sub, rel, obj)
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0)  # number of predictions
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0)

                for k in [1, 3, 10]:
                    results[f'hits@{k}'] = torch.numel(ranks[ranks <= k]) + results.get(f'hits@{k}', 0)
        return results

    def save_model(self, path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.
        :param path: path where the model is saved
        :return:
        """
        state = {
            'model': self.model.state_dict(),
            'best_val': self.best_val_results,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, path)

    def load_model(self, path):
        """
        Function to load a saved model
        :param path: path where model is loaded
        :return:
        """
        state = torch.load(path)
        self.best_val_results = state['best_val']
        self.best_val_mrr = self.best_val_results['mrr']
        self.best_epoch = state['best_epoch']
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])

    def build_graph(self):
        g = dgl.DGLGraph()
        g.add_nodes(self.num_ent)
        g.add_edges(self.train_data[:, 0], self.train_data[:, 2])
        g.add_edges(self.train_data[:, 2], self.train_data[:, 0])
        return g

    def get_data_iter(self):
        """
        get data loader for train, valid and test section
        :return: dict
        """

        def get_data_loader(dataset_class, split):
            return DataLoader(
                dataset_class(self.triplets[split], self.num_ent, self.p),
                batch_size=self.p.batch_size,
                shuffle=True,  # 测试应该使用false
                num_workers=self.p.num_workers
            )

        return {
            'train': get_data_loader(TrainDataset, 'train'),
            'valid_head': get_data_loader(TestDataset, 'valid_head'),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail'),
            'test_head': get_data_loader(TestDataset, 'test_head'),
            'test_tail': get_data_loader(TestDataset, 'test_tail')
        }

    def get_edge_dir_and_norm(self):
        """
        :return: edge_type: indicates type of each edge: [E]
        """
        in_deg = self.g.in_degrees(range(self.g.number_of_nodes())).float().numpy()
        norm = in_deg ** -0.5
        norm[np.isinf(norm)] = 0
        # self.g.ndata['xxx'] = norm
        self.g.ndata['xxx'] = torch.tensor(norm)
        self.g.apply_edges(lambda edges: {'xxx': edges.dst['xxx'] * edges.src['xxx']})  # 为什么in&out相乘
        norm = self.g.edata.pop('xxx').squeeze().to(self.device)
        edge_type = torch.tensor(np.concatenate([self.train_data[:, 1], self.train_data[:, 1] + self.num_rels])).to(
            self.device)
        return edge_type, norm

    def get_model(self):
        # model = VQGCN(num_ent=self.num_ent, num_rel=self.num_rels, gcn_layers=self.p.gcn_layers,
        model = VQGCN_MLP(num_ent=self.num_ent, num_rel=self.num_rels, gcn_layers=self.p.gcn_layers,
                      tf_layers=self.p.tf_layers, init_dim=self.p.init_dim,
                      gcn_dim=self.p.gcn_dim, edge_type=self.edge_type, edge_norm=self.edge_norm,
                      gcn_drop=self.p.gcn_drop, act=self.p.act, opn='mult', seq_len=self.p.seq_len, num_code=self.p.num_code, att_head=self.p.att_head)
        model.to(self.device)
        return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', default='test_run', help='Set run name for saving/restoring models')
    parser.add_argument('--data', dest='dataset', default='data/FB15K-237N', help='Dataset to use, default: FB15k-237')  # FB15k-237  wn18rr  FB15K-237N  CoDeX-S
    parser.add_argument('--opn', dest='opn', default='mult', help='Composition Operation to be used in CompGCN')  # corr
    parser.add_argument('--act', dest='act', default='tanh', help='activation function')  # tanh relu

    parser.add_argument('--batch', dest='batch_size', default=1024, type=int, help='Batch size')  # 256
    parser.add_argument('--gpu', type=int, default=7, help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
    parser.add_argument('--epoch', dest='max_epochs', type=int, default=800, help='Number of epochs')  # 500
    parser.add_argument('--l2', type=float, default=1e-8, help='L2 Regularization for Optimizer')  # 1e-8
    parser.add_argument('--lr', type=float, default=0.0005, help='Starting Learning Rate')  # 0.001 0.0005 0.0001  0005比较好
    parser.add_argument('--lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of processes to construct batches')
    parser.add_argument('--seed', dest='seed', default=12345, type=int, help='Seed for randomization')

    parser.add_argument('--restore', dest='restore', action='store_true',
                        help='Restore from the previously saved model')

    parser.add_argument('--init_dim', dest='init_dim', default=200, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('--gcn_dim', dest='gcn_dim', default=200, type=int, help='Number of hidden units in GCN')
    parser.add_argument('--seq_len', dest='seq_len', default=16, type=int, help='Number of VQ sequence')
    parser.add_argument('--num_code', dest='num_code', default=1024, type=int, help='Number of VQ codebook')  # 2048
    parser.add_argument('--gcn_layers', dest='gcn_layers', default=2, type=int, help='Number of GCN Layers to use')  # 1
    parser.add_argument('--tf_layers', dest='tf_layers', default=1, type=int, help='Number of Transformer Layers to use')
    parser.add_argument('--att_head', dest='att_head', default=1, type=int, help='Number of attention head in Transformer')
    parser.add_argument('--gcn_drop', dest='gcn_drop', default=0.2, type=float, help='Dropout to use in GCN Layer')  # 0.1 0.3效果好
    parser.add_argument('--vq_weight', dest='vq_weight', default=1.0, type=float, help='VQ codebook weight in the loss')


    args = parser.parse_args()
    if not args.restore:
        # args.name = time.strftime('%Y_%m_%d') + '_' + time.strftime(
        #     '%H:%M:%S') + '-' + args.opn
        dataset = args.dataset.split('/')[-1]
        save_name = f'{dataset}_{args.seq_len}_{args.num_code}' + time.strftime('%Y_%m_%d') + '_' + time.strftime(
            '%H:%M:%S')  # GCN 2
        args.name = save_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    runner = Runner(args)
    runner.fit()


