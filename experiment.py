from sklearn.cluster import KMeans
import numpy as np
from model import NLPClassifier
import torchvision
from torchvision import transforms as transforms
import torch
from torch.utils.data import DataLoader as Loader
from utils import SpreadSheetNLPCustomDataset

class Experiment(object):
    def __init__(self, config: dict):
        self.classifier = NLPClassifier(config)

    def _run(self, dataset, config: dict):
        split = self._preprocessing(dataset, True)
        init_epoch = self.classifier.curr_epoch
        loaders = [Loader(ds, self.classifier.bs, shuffle=True, num_workers=4) for ds in split]
        score, k, indices = self._features_selection(loaders[0])
        print("features selected, optimal model score = ", score)
        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            f1_train, f1_val, acc_train, acc_val, loss_train, loss_val = self.classifier._run_epoch(loaders, indices, k)
            print("Epoch: {} | f1 Train: {} | f1 Val  {} | Training Accuracy: {} | Validation Accuracy: {} | Training Loss: {} | Validation Loss: {} | ".format(self.classifier.curr_epoch, f1_train, f1_val, acc_train, acc_val, loss_train, loss_val))
            self.classifier.writer.add_scalar("Training Accuracy", acc_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Accuracy",acc_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Training Loss",loss_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Loss",loss_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Train",f1_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Val",f1_val, self.classifier.curr_epoch)
            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
            loaders = [Loader(ds, self.classifier.bs, shuffle=True, num_workers=4) for ds in split]
        print("Testing:...")
        print(self.classifier._validate(loaders[2], indices, k))
        print("\nRun Complete.")

    def _preprocessing(self, directory, train):
        dataSetFolder = SpreadSheetNLPCustomDataset(directory, self.classifier.tokenizer)
        #@TODO add features selection here
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2           
            splits = torch.utils.data.random_split(dataSetFolder, [trainingValidationDatasetSize, testDatasetSize, testDatasetSize])
            #split_names = ['train', 'validation', 'test']
            #classes = list(dataSetFolder.labels.items())
            #print(classes)
            #distributions = {split_names[i]: {k: len(list(filter(lambda x: x["labels"]==v, splits[i]))) for k,v in classes} for i in range(len(splits))}
            #print(distributions)
            return splits
        return dataSetFolder
    
    def _features_selection(self, loader):
        data = next(iter(loader))
        X = data["input_ids"].cpu().numpy()
        K = 2
        score = float("-inf")
        i = -1
        t_score = [self.classifier._score(loader, [i for i in range(X.shape[1])], i)]
        Z = torch.tensor(X.T)
        while max(t_score) != score:
            X = next(iter(loader))["input_ids"].cpu().numpy()
            Z = torch.tensor(X.T)
            if max(t_score) < score:
                print(score, K-2, i)
                t_score.append(score)
            kmeans = KMeans(K, init="k-means++")
            indices = torch.tensor(kmeans.fit_predict(Z))
            clusters = {i: Z[indices==i] for i in range(K)}
            avg = sum(list(map(lambda c: len(c),list(clusters.values()))))/K
            clusters = list(filter(lambda k: len(clusters[k])>=avg,list(clusters.keys())))
            l = list(map(lambda idx: (idx, self.classifier._score(loader, indices, idx)), clusters))
            s = max(list(map(lambda l_: l_[1],l)))
            if float('nan') in list(map(lambda l_: l_[1],l)) or s == float('nan') or max(t_score) > s:
                continue
            score = s
            l = list(filter(lambda a_: a_[1] == score, l))
            try:
                i = l[0]
            except:
                continue
            K += 2
        return score, i, indices