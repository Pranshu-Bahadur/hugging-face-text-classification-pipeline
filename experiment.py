from kmeans_pytorch import kmeans
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
            metrics_train = self.classifier.run_epoch_step(loaders[0], "train", e_num=self.classifier.curr_epoch)
            print(f"\nValidation step @ {self.classifier.curr_epoch}:\n# of samples = {len(splits[1])}\n")
            with torch.no_grad():
                metrics_validation = self.classifier.run_epoch_step(loaders[1], "validation", e_num=self.classifier.curr_epoch)
            print(f"----Results at {self.classifier.curr_epoch}----\n")
            print(f"\nFor train split:\n{metrics_train}\n")
            print(f"\nFor validation split:\n{metrics_validation}\n")
            if self.classifier.curr_epoch%self.classifier.save_interval==0:
                self.classifier._save(self.classifier.save_directory, "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
        print(f"\nTesting model trained for {self.classifier.curr_epoch}:\n# of samples = {len(splits[2])}\n")
        metrics_test = self.classifier.run_epoch_step(loaders[2], "test", e_num=self.classifier.curr_epoch)
        print(f"\nFinal Results:\n{metrics_test}\n")
        print("\nRun Complete.\n\n")

    def _preprocessing(self, train):
        dataSetFolder = self.classifier.dataset               
        print("\n\nRunning K-means for outlier detection...\n\n")
        X = torch.tensor(torch.tensor(dataSetFolder.encodings["input_ids"])).cuda()
        X = X.view(X.size(0), -1)
        cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=8, device=torch.device('cuda:0'))
        _, indices = torch.topk(torch.tensor([(cluster_ids_x==i).nonzero().size(0) for i in range(8)]), 7)
        indices = torch.cat([(cluster_ids_x==i).nonzero() for i in indices], dim=0).view(-1).tolist()
        print(f"\n\nResult of k-means: {len(indices)} of {X.size(0)} samples remain, taken from top 7 cluster(s) according to mode.\n\n")
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2
            splits = [trainingValidationDatasetSize, testDatasetSize, testDatasetSize]
            diff = len(dataSetFolder) - sum(splits)
            splits.append(diff)
            splits = torch.utils.data.dataset.random_split(dataSetFolder, splits)
            return splits
        return dataSetFolder
