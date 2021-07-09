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
        scores = [-1]
        K = 2
        #scores, k, indices = self._features_selection(loaders[0], K, max(scores))
        #print("Features selection with K {} complete:".format(K))
        #self.classifier.criterion = torch.nn.CrossEntropyLoss(weight=weights).cuda()
        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            print("Epoch {} Features selection with K {}:".format(self.classifier.curr_epoch+1, K), "--------------------")
            score_, k_, indices_ = self._features_selection(loaders[0], K, max(scores))
            if max(scores) < score_:
                #K = K // 2 if K > 2 else 8
                print("Better score - updating features.")
                score, k, indices = score_, k_, indices_
                scores.append(score)
            print("Epoch {} Training Model based of newly selected features:".format(self.classifier.curr_epoch+1), "--------------------")
            
            f1_train, f1_val, acc_train, acc_val, loss_train, loss_val = self.classifier._run_epoch(loaders, indices, k)
            print("Epoch {} Results: | Features Score {} | f1 Train: {} | f1 Val  {} | Training Accuracy: {} | Validation Accuracy: {} | Training Loss: {} | Validation Loss: {} | ".format(self.classifier.curr_epoch, score, f1_train, f1_val, acc_train, acc_val, loss_train, loss_val))
            self.classifier.writer.add_scalar("Training Accuracy", acc_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Accuracy",acc_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Training Loss",loss_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Loss",loss_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Train",f1_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Val",f1_val, self.classifier.curr_epoch)
            loaders[0] = Loader(split[0], self.classifier.bs, shuffle=True, num_workers=4)
            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
        print("Testing:...")
        print(self.classifier._validate(loaders[2], indices, k))
        print("\nRun Complete.")

    def _preprocessing(self, directory, train):
        dataSetFolder = SpreadSheetNLPCustomDataset(directory, self.classifier.tokenizer, self.classifier.library, self.classifier.long)
        #@TODO add features selection here
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2           
            splits = torch.utils.data.random_split(dataSetFolder, [trainingValidationDatasetSize, testDatasetSize, testDatasetSize])
            #split_names = ['train', 'validation', 'test']
            #classes = list(dataSetFolder.labels.items())
            #self.classifier.writer.add_text("Classes:",f'{classes}')
            #distributions = {split_names[i]: {k: len(list(filter(lambda x: x["labels"]==v, splits[i]))) for k,v in classes} for i in range(len(splits))}
            #self.classifier.writer.add_text("Run distribution:",f'{distributions}')
            #total = sum(list(distributions[0].values()))
            #weights = torch.tensor(list(distributions["train"].values())).float()
            #weights -= weights.min().item()
            #weights /= weights.max().item()
            weights = []
            return splits, weights
        return dataSetFolder
    #@TODO...improve this...
    def _features_selection(self, loader, K, score):
        #self.classifier.model.train()
        data = next(iter(loader))
        X = data["input_ids"].cpu().numpy() #if self.classifier.library != "timm" else np.concatenate(tuple([data["input_ids"].view(self.classifier.bs, -1).cpu().numpy() for data in loader][:-1]), axis=0)
        #X = 
        #X = torch.tensor(X).view(-1, 512).cpu().numpy()
        i = -1
        t_score = [1e-4]
        iterations = 0
        indices = []
        #Z = torch.tensor(X.T)
        memoisation = {}
        memoisation[score] = (indices,i)
        while max(list(memoisation.keys())) != score:
            X = next(iter(loader))["input_ids"].cpu().numpy()# if self.classifier.library != "timm" else X
            Z = torch.tensor(X.T)
            iterations += 1
            if score >= max(list(memoisation.keys())):
                print(f"Updating...at {iterations}, done for {K} clusters, with score = {score}")
                if score in list(memoisation.keys()):
                    print(f"Convergence at {iterations}, done for {K} clusters, with score = {score}")
                    return score, i, indices
                memoisation[score] = (indices, i)
            kmeans = KMeans(K, init="k-means++")
            indices = torch.tensor(kmeans.fit_predict(Z))
            #clusters = {i: Z[indices==i].float().cuda() for i in range(K)}
            #big_c = torch.mean(torch.stack(list(map(lambda c: torch.mean(c),list(clusters.values())))), -1)
            #clusters = list(filter(lambda k: torch.mean(clusters[k])>=big_c, list(clusters.keys())))
            l = list(map(lambda idx: (idx, self.classifier._score(loader, indices, idx)), [i for i in range(K)]))#clusters
            l = list(filter(lambda a_: float('nan') != a_[1] and max(list(memoisation.keys())) <= a_[1], l))
            if len(l) == 0:
                score = -100
                print("Naan bread detected...")
                continue
            s = max(list(map(lambda l_: l_[1],l)))
            if s == 0 or s == float('nan') or s == float('-inf') or s == float('inf'):
                print("max is 0...")
                continue
            score = s
            if score == s:
                print(f"{iterations}: K = {K} score = {score}")
                #K+=2
            try:
                i = l[0][0]
            except:
                print("Random unidentified error...")
                continue
        return score, i, indices