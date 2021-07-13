from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
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
    def __init__(self, config: dict):
        #TODO config.label2ids = 
        self.classifier = NLPClassifier(config)

    def _run(self, dataset, config: dict):
        dataset, splits = self._preprocessing(dataset, True)#, indices, Y_ 
        init_epoch = self.classifier.curr_epoch
        #weights.reverse()
        #self.classifier.criterion.weight = torch.tensor(weights).float().cuda()
        #random.shuffle(indices)
        #dist = [Y_[indices[:splits[0]]].size(0) for i in Y_.unique()]
        #beta = 0.999
        #effective_num = 1.0 - np.power(beta, dist)
        #weights = (1 - beta)/np.array(effective_num)
        #weights = weights/np.sum(weights)*self.classifier.nc
        #self.classifier.criterion = nn.CrossEntropyLoss(weight= torch.tensor(weights).float().cuda()).cuda()
        
        loaders = [Loader(splits[0], self.classifier.bs, shuffle=True, num_workers=4),#, sampler=#indices[:splits[0]]),
        Loader(splits[1], self.classifier.bs, shuffle=True, num_workers=4),#, sampler=indices[splits[1]:]),
        Loader(splits[2], self.classifier.bs, shuffle=True, num_workers=4),#, sampler=indices[splits[1]+splits[2]:]),
        ]
        
        print("Dataset has been preprocessed and randomly split.\nRunning training loop...\n")
        print("\nRunning dimensoniality reduction...\nRunning training loop...\n")
        #self.classifier._k_means_approximation_one_step(loaders[0])
        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            logs = self.classifier._run_epoch(loaders[:-1])
            print(f"Epoch {self.classifier.curr_epoch} Results {logs}\n\n")
            """
            f1_train, f1_val, acc_train, acc_val, loss_train, loss_val = self.classifier._run_epoch(loaders)
            print("Epoch {} Results: | Features Score {} | f1 Train: {} | f1 Val  {} | Training Accuracy: {} | Validation Accuracy: {} | Training Loss: {} | Validation Loss: {} | ".format(self.classifier.curr_epoch, self.classifier.score, f1_train, f1_val, acc_train, acc_val, loss_train, loss_val))
            self.classifier.writer.add_scalar("Training Accuracy", acc_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Accuracy",acc_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Training Loss",loss_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Loss",loss_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Train",f1_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Val",f1_val, self.classifier.curr_epoch)
            #loaders[0] = Loader(split[0], self.classifier.bs, shuffle=True, num_workers=4)
            """
            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
        print("Testing:...")
        print(self.classifier._validate(loaders[2]))
        print("\nRun Complete.")

    def _preprocessing(self, directory, train):
        dataSetFolder = self.classifier.dataset       
        #loader = Loader(dataSetFolder, self.classifier.bs, shuffle=False, num_workers=4)
        #print("\n\nRunning K-means for outlier detection...\n\n")
        #X = torch.tensor(torch.tensor(dataSetFolder.encodings["input_ids"])).cuda()
        #X = X.view(X.size(0), -1)
        #Y = torch.tensor(np.asarray(dataSetFolder.dataset["type"].values))
        #Y = Y.view(Y.size(0), -1)
        #XY = torch.cat((X,Y), dim=1).cuda()
        #cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=8, device=torch.device('cuda:0'))
        #topk, indices = torch.topk(torch.tensor([(cluster_ids_x==i).nonzero().size(0) for i in range(8)]), 7)#torch.tensor([torch.mean(cluster_centers[i].float()).float() for i in range(8)]),2)#torch.tensor([(cluster_ids_x==i).nonzero().size(0) for i in range(8)]), 1)
        #indices = torch.cat([(cluster_ids_x==i).nonzero() for i in indices], dim=0).view(-1).tolist()
        #print(f"\n\nResult of k-means: {len(indices)} samples remain, taken from top 7 cluster(s)\n\n")
        #X_ = X[indices]
        #dist = [Y_[indices].cpu().size(0) for i in Y_.unique()]

        #@TODO add features selection here
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2
            diff = len(dataSetFolder) - sum([trainingValidationDatasetSize, testDatasetSize, testDatasetSize])
            splits = [trainingValidationDatasetSize, testDatasetSize, testDatasetSize]
            splits = torch.utils.data.dataset.random_split(dataSetFolder, splits)
            total = sum(list(dataSetFolder.distribution.values()))
            return dataSetFolder ,splits#, indices, Y
        return dataSetFolder