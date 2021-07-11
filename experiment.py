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
        dataset, splits, indices = self._preprocessing(dataset, True)
        init_epoch = self.classifier.curr_epoch
        random.shuffle(indices)
        loaders = [Loader(dataset, self.classifier.bs, shuffle=False, num_workers=4, sampler=indices[:splits[0]]),
        Loader(dataset, self.classifier.bs, shuffle=False, num_workers=4, sampler=indices[splits[1]:]),
        Loader(dataset, self.classifier.bs, shuffle=False, num_workers=4, sampler=indices[splits[1]+splits[2]:]),
        ]
        print("Dataset has been preprocessed and randomly split.\nRunning training loop...\n")
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
        dataSetFolder = SpreadSheetNLPCustomDataset(directory, self.classifier.tokenizer, self.classifier.library, [])
        #loader = Loader(dataSetFolder, self.classifier.bs, shuffle=False, num_workers=4)
        print("\n\nRunning K-means for outlier detection...\n\n")
        X = torch.tensor(torch.tensor(dataSetFolder.encodings["input_ids"])).cuda()
        X = X.view(X.size(0), -1)
        cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=8, device=torch.device('cuda:0'))
        topk, indices = torch.topk(torch.cat([(cluster_ids_x==i).nonzero() for i in range(8)]), 7)
        indices = torch.cat([(cluster_ids_x==i).nonzero() for i in indices], dim=0).view(-1).tolist()
        print(f"\n\nResult of k-means: {len(indices)} samples remain, taken from top 12 clusters\n\n")

        #@TODO add features selection here
        if train:
            trainingValidationDatasetSize = int(0.6 * len(indices))
            testDatasetSize = int(len(indices) - trainingValidationDatasetSize) // 2
            diff = len(dataSetFolder) - sum([trainingValidationDatasetSize, testDatasetSize, testDatasetSize])
            splits = [trainingValidationDatasetSize, testDatasetSize, testDatasetSize]
            weights = []
            return dataSetFolder ,splits, indices
        return dataSetFolder