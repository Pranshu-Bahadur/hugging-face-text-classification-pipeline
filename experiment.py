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
from torch.utils.data import WeightedRandomSampler

class Experiment(object):
    def __init__(self, config):
        self.classifier = NLPClassifier(config)
        self.class_weights = np.array([])

    def _run(self):
        splits = self._preprocessing(True)
        init_epoch = self.classifier.curr_epoch
        print("Computing Weighted Random Sampler")
        sampler_loader = Loader(splits[0], self.classifier.bs, shuffle=False, num_workers=4)
        target = torch.cat([data["labels"] for data in sampler_loader])
        samples_weight = np.array([self.class_weights[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        target = torch.from_numpy(target).long()
        train_loader = Loader(splits[0], self.classifier.bs, shuffle=False, num_workers=4, sampler=sampler)
        loaders = [Loader(split, self.classifier.bs, shuffle=True, num_workers=4) for split in splits[1:]]
        print("Dataset has been preprocessed and randomly split.\nRunning training loop...\n")
        while (self.classifier.curr_epoch < init_epoch + self.classifier.final_epoch):
            self.classifier.curr_epoch +=1
            print(f"----Running epoch {self.classifier.curr_epoch}----\n")
            print(f"Training step @ {self.classifier.curr_epoch}:\n# of samples = {len(splits[0])}\n")
            metrics_train = self.classifier.run_epoch_step(train_loader, "train")
            print(f"\nValidation step @ {self.classifier.curr_epoch}:\n# of samples = {len(splits[1])}\n")
            with torch.no_grad():
                metrics_validation = self.classifier.run_epoch_step(loaders[0], "validation")
            print(f"----Results at {self.classifier.curr_epoch}----\n")
            print(f"\nFor train split:\n{metrics_train}\n")
            print(f"\nFor validation split:\n{metrics_validation}\n")
            if self.classifier.curr_epoch%self.classifier.save_interval==0:
                self.classifier._save(self.classifier.save_directory, "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
        print(f"\nTesting model trained for {self.classifier.curr_epoch}:\n# of samples = {len(splits[2])}\n")
        with torch.no_grad():
            metrics_test = self.classifier.run_epoch_step(loaders[1], "test")
        print(f"\nFinal Results:\n{metrics_test}\n")
        print("\nRun Complete.\n\n")

    #TODO I give up on using elbow for global minimum...I'm happy with local. That's what the method is for...
    def finding_k(self, X, n):
        X = X.view(X.size(0), -1)
        m_dict = {}
        differences = [0]
        for k in range(2, n+1):
            cluster_ids, centers  = kmeans(X=X, num_clusters = k, device=torch.device('cuda'))
            curr_inertia = sum([torch.sum((1/(2*i+1))*pairwise_distance(X[cluster_ids==i], centers[i]), 0).cpu().item() for i in range(k)])/1e+5
            self.classifier.writer.add_scalar("Inertia",curr_inertia, k)
            if k!=2:
                prev_inertias = list(m_dict.keys())
                difference = int(sum(prev_inertias)/len(prev_inertias)) - int((sum(prev_inertias) - curr_inertia)/len(prev_inertias)+1)
                if len(differences)>2 and differences[-1] < difference: #abs(max(differences) - difference)
                    print(f"Elbow at {k-1}")
                    break
                differences.append(difference)
            m_dict[curr_inertia] = {"k": k, "cluster_ids": cluster_ids, "centers": centers}
        result = m_dict[list(m_dict.keys())[-1]]
        return result["k"], result["cluster_ids"], result["centers"]

    def distribution(self, split, num_classes):
        loader = Loader(split, self.classifier.bs, shuffle=False, num_workers=4)
        train_Y = torch.cat([data["labels"] for data in loader])
        train_split_dist = [(train_Y==i).sum().cpu().item() for i in range(num_classes)]
        return train_split_dist
   
    def weight_calc(self, distribution, beta):
        imb_weights = []
        for num_samples in distribution:
            effective_num = 1.0 - np.power(beta, num_samples)
            weights = (1.0 - beta) / effective_num
            imb_weights.append(weights)
        imb_weights = [weights/sum(imb_weights) * self.classifier.nc for weights in imb_weights]
        #imb_weights.reverse()
        return np.array(imb_weights)#torch.FloatTensor(imb_weights)


    def _preprocessing(self, train):
        dataSetFolder = self.classifier.dataset
        print("\n\nRunning K-means for outlier detection...\n\n")
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2
            splits = [trainingValidationDatasetSize, testDatasetSize, testDatasetSize]
            diff = len(dataSetFolder) - sum(splits)
            if diff > 0:
                splits.append(diff)
            splits = torch.utils.data.dataset.random_split(dataSetFolder, splits)
            k_means_loader = Loader(splits[0], self.classifier.bs, shuffle=True, num_workers=4)
            X = torch.cat([x["input_ids"] for x in k_means_loader]).cuda()
            best_k, cluster_ids_x, cluster_centers = self.finding_k(X, X.size(1))
            print(best_k, cluster_ids_x, cluster_centers)
            _, indices = torch.topk(torch.tensor([(cluster_ids_x==i).nonzero().size(0) for i in range(best_k)]), best_k//2)
            indices = torch.cat([(cluster_ids_x==i).nonzero() for i in indices], dim=0).view(-1).tolist()
            print(f"\n\nResult of k-means on {best_k} clusters: {len(indices)} of {X.size(0)} samples remain, taken from top {best_k//2} cluster(s) according to mode.\n\n")
            splits[0] = torch.utils.data.dataset.Subset(splits[0],indices)
            train_split_dist = self.distribution(splits[0], 16)
            self.class_weights = self.weight_calc(train_split_dist, 0.9).cuda() #TODO find proper betas value.
            return splits[:-1] if diff > 0 else splits
        return dataSetFolder