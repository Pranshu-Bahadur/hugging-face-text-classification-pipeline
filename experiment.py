from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
import numpy as np
from model import NLPClassifier
import torchvision
from torchvision import transforms as transforms
import torch
from torch.utils.data import DataLoader as Loader
from utils import SpreadSheetNLPCustomDataset
import random

class Experiment(object):
    def __init__(self, config: dict):
        self.classifier = NLPClassifier(config)

    def _run(self, dataset, config: dict):
        split, weights = self._preprocessing(dataset, True)
        init_epoch = self.classifier.curr_epoch
        loaders = [Loader(data, self.classifier.bs, shuffle=True, num_workers=4) for data in split]
        
        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            f1_train, f1_val, acc_train, acc_val, loss_train, loss_val = self.classifier._run_epoch(loaders)
            print("Epoch {} Results: | Features Score {} | f1 Train: {} | f1 Val  {} | Training Accuracy: {} | Validation Accuracy: {} | Training Loss: {} | Validation Loss: {} | ".format(self.classifier.curr_epoch, self.classifier.score, f1_train, f1_val, acc_train, acc_val, loss_train, loss_val))
            self.classifier.writer.add_scalar("Training Accuracy", acc_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Accuracy",acc_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Training Loss",loss_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Loss",loss_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Train",f1_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Val",f1_val, self.classifier.curr_epoch)
            #loaders[0] = Loader(split[0], self.classifier.bs, shuffle=True, num_workers=4)
            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
        print("Testing:...")
        print(self.classifier._validate(loaders[2]))
        print("\nRun Complete.")

    def _preprocessing(self, directory, train):
        dataSetFolder = SpreadSheetNLPCustomDataset(directory, self.classifier.tokenizer, self.classifier.library)
        #loader = Loader(dataSetFolder, self.classifier.bs, shuffle=False, num_workers=4)
        print("Running K-means for outlier detection...")
        X = torch.tensor(torch.tensor(dataSetFolder.encodings["input_ids"])).cuda()
        X = X.view(X.size(0), -1)
        cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=16, device=torch.device('cuda:0'))
        print(torch.topk(cluster_centers, 8, dim=0))
        topk, indices = torch.topk(torch.mean(cluster_centers, dim=-1), 8)
        print(indices)
        print("Result of k-means:",topk, cluster_centers[indices], cluster_ids_x)
        print(torch.logical_or([cluster_ids_x==i for i in indices]).size(0))
        dataSetFolder = dataSetFolder[torch.logical_or(torch.cat([cluster_ids_x==i for i in indices], dim=-1)).nonzero(as_tuple=True).tolist()]


        #@TODO add features selection here
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2
            diff = len(dataSetFolder) - sum([trainingValidationDatasetSize, testDatasetSize, testDatasetSize])
            print(len(dataSetFolder), diff)
            splits = torch.utils.data.random_split(dataSetFolder, [trainingValidationDatasetSize, testDatasetSize, testDatasetSize])
            weights = []
            print("Data set has been randomly split and preprocessed")
            return splits, weights
        return dataSetFolder