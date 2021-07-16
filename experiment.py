from kmeans_pytorch import kmeans, pairwise_distance
import numpy as np
from model import NLPClassifier
import torchvision
from torchvision import transforms as transforms
import torch
from torch import nn as nn
from torch.utils.data import DataLoader as Loader
from utils import SpreadSheetNLPCustomDataset
import random

class Experiment(object):
    def __init__(self, config):
        self.classifier = NLPClassifier(config)

    def _run(self):
        splits = self._preprocessing(True)
        init_epoch = self.classifier.curr_epoch
        loaders = [Loader(split, self.classifier.bs, shuffle=True, num_workers=4) for split in splits]
        print("Dataset has been preprocessed and randomly split.\nRunning training loop...\n")
        while (self.classifier.curr_epoch < init_epoch + self.classifier.final_epoch):
            self.classifier.curr_epoch +=1
            print(f"----Running epoch {self.classifier.curr_epoch}----\n")
            print(f"Training step @ {self.classifier.curr_epoch}:\n# of samples = {len(splits[0])}\n")
            metrics_train = self.classifier.run_epoch_step(loaders[0], "train")
            print(f"\nValidation step @ {self.classifier.curr_epoch}:\n# of samples = {len(splits[1])}\n")
            with torch.no_grad():
                metrics_validation = self.classifier.run_epoch_step(loaders[1], "validation")
            print(f"----Results at {self.classifier.curr_epoch}----\n")
            print(f"\nFor train split:\n{metrics_train}\n")
            print(f"\nFor validation split:\n{metrics_validation}\n")
            if self.classifier.curr_epoch%self.classifier.save_interval==0:
                self.classifier._save(self.classifier.save_directory, "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
        print(f"\nTesting model trained for {self.classifier.curr_epoch}:\n# of samples = {len(splits[2])}\n")
        with torch.no_grad():
            metrics_test = self.classifier.run_epoch_step(loaders[2], "test")
        print(f"\nFinal Results:\n{metrics_test}\n")
        print("\nRun Complete.\n\n")

    def finding_k(self, X, n):
        X = X.view(X.size(0), -1)
        m_dict = {}
        differences = []
        for k in range(2, n+1):
            cluster_ids, centers  = kmeans(X=X, num_clusters = k, device=torch.device('cuda'))
            print(cluster_ids)
            curr_inertia = torch.sum((1/2*i+1)*sum([torch.sum((X[cluster_ids==i].cpu() - centers[i].cpu())**2, dim=0) for i in range(k)]), dim=-1)
            print(curr_inertia)
            if k!=2:
                highest_inertia_key = max(list(m_dict.keys()))
                prev_inertia_key = list(m_dict.keys())[-1]
                m = lambda y1,x1: (curr_inertia - y1)/(k - x1)
                difference = int(((m(highest_inertia_key, m_dict[highest_inertia_key]["k"])) - (m(prev_inertia_key, m_dict[prev_inertia_key]["k"]))))
            if k!=2 and differences[-1] == difference:
                print("Elbow?")
                break
            m_dict[curr_inertia] = {"k": k, "cluster_ids": cluster_ids, "centers": centers}
            differences.append(difference)
        result = m_dict[list(m_dict.keys())[-1]]
        return result["k"], result["cluster_ids"], result["centers"]

    def distribution(self, split, num_classes):
        loader = Loader(split, self.classifier.bs, shuffle=True, num_workers=4)
        train_Y = torch.cat([data["labels"] for data in loader])
        train_split_dist = [(train_Y==i).sum().cpu().item() for i in range(num_classes)]
        return train_split_dist
   
    def weight_calc(self, distribution, beta):
        imb_weights = []
        for num_samples in distribution:
            effective_num = 1.0 - np.power(beta, num_samples)
            weights = (1.0 - beta) / effective_num
            imb_weights.append(weights)
        imb_weights = [weights/sum(imb_weights) * 16 for weights in imb_weights]
        return torch.FloatTensor(imb_weights)


    def _preprocessing(self, train):
        dataSetFolder = self.classifier.dataset
        print("\n\nRunning K-means for outlier detection...\n\n")
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2
            splits = [trainingValidationDatasetSize, testDatasetSize, testDatasetSize]
            diff = len(dataSetFolder) - sum(splits)
            splits.append(diff)
            splits = torch.utils.data.dataset.random_split(dataSetFolder, splits)
            k_means_loader = Loader(splits[0], self.classifier.bs, shuffle=True, num_workers=4)
            X = torch.cat([x["input_ids"] for x in k_means_loader]).cuda()
            best_k, cluster_ids_x, cluster_centers = self.finding_k(X, 20)
            print(best_k, cluster_ids_x, cluster_centers)
            _, indices = torch.topk(torch.tensor([(cluster_ids_x==i).nonzero().size(0) for i in range(best_k)]), 2)
            indices = torch.cat([(cluster_ids_x==i).nonzero() for i in indices], dim=0).view(-1).tolist()
            print(f"\n\nResult of k-means on {best_k} clusters: {len(indices)} of {X.size(0)} samples remain, taken from top 2 cluster(s) according to mode.\n\n")
            splits[0] = torch.utils.data.dataset.Subset(splits[0],indices)
            train_split_dist = self.distribution(splits[0], 16)
            self.classifier.criterion.weight = self.weight_calc(train_split_dist, 0.9).cuda() #TODO find proper betas value.
            return splits[:-1]
        return dataSetFolder