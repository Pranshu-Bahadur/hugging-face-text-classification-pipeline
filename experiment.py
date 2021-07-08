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
        split = self._preprocessing(dataset, True)
        init_epoch = self.classifier.curr_epoch
        loaders = [Loader(data, self.classifier.bs, shuffle=True, num_workers=4) for data in split]
        score = float('-inf')
        K = 512
        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            K = K // 2 if K != 2 else 256
            print("Epoch {} Features selection with K {}:".format(self.classifier.curr_epoch+1, K), "--------------------")
            score_, k_, indices_ = self._features_selection(loaders[0], K)
            if score < score_:
                print("Better score updating features.")
                score, k, indices = score_, k_, indices_
            print("Epoch {} Training Model based of newly selected features:".format(self.classifier.curr_epoch+1), "--------------------")
            f1_train, f1_val, acc_train, acc_val, loss_train, loss_val = self.classifier._run_epoch(loaders, indices, k)
            print("Epoch {} Results: | Features Score {} | f1 Train: {} | f1 Val  {} | Training Accuracy: {} | Validation Accuracy: {} | Training Loss: {} | Validation Loss: {} | ".format(self.classifier.curr_epoch, score, f1_train, f1_val, acc_train, acc_val, loss_train, loss_val))
            self.classifier.writer.add_scalar("Training Accuracy", acc_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Accuracy",acc_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Training Loss",loss_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Loss",loss_val, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Train",f1_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Val",f1_val, self.classifier.curr_epoch)
            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
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
            dataSetFolder = torch.utils.data.random_split(dataSetFolder, [trainingValidationDatasetSize, testDatasetSize, testDatasetSize])[0]
            #split_names = ['train', 'validation', 'test']
            #classes = list(dataSetFolder.labels.items())
            #self.classifier.writer.add_text("Classes:",f'{classes}')
            #distributions = {split_names[i]: {k: len(list(filter(lambda x: x["labels"]==v, splits[i]))) for k,v in classes} for i in range(len(splits))}
            #self.classifier.writer.add_text("Run distribution:",f'{distributions}')
            return splits
        return dataSetFolder
    #@TODO...improve this...
    def _features_selection(self, loader, K):
        data = next(iter(loader))
        X = data["input_ids"].cpu().numpy()
        #np.concatenate(tuple([data["input_ids"].cpu().numpy() for data in loader]), axis=0)
        score = float("-inf")
        i = -1
        t_score = [1e-4]
        iterations = 0
        indices = []
        while max(t_score) != score:
            iterations += 1
            Z = torch.tensor(X.T)
            if max(t_score) <= score:
                print(f"Updating...at {iterations}, done for {K} clusters, with score = {score}")
                t_score.append(score)
                if len(list(filter(lambda x: x == max(t_score), t_score))) > 1:
                    print(f"Convergence at {iterations}, done for {K} clusters, with score = {score}")
                    return score, i, indices
                #K += 2
            kmeans = KMeans(K, init="k-means++")
            indices = torch.tensor(kmeans.fit_predict(Z))
            #clusters = {i: Z[indices==i] for i in range(K)}
            #big_c = max(list(map(lambda c: len(c),list(clusters.values()))))
            #clusters = list(filter(lambda k: len(clusters[k])==big_c, list(clusters.keys())))
            l = list(map(lambda idx: (idx, self.classifier._score(data, indices, idx)), [i for i in range(K)]))#clusters
            l = list(filter(lambda a_: float('nan') != a_[1], l))
            if len(l) == 0:
                print("Naan bread detected...Reshuffling.")
                data = next(iter(loader))
                X = data["input_ids"].cpu().numpy()
                #K += 2
                continue
            s = max(list(map(lambda l_: l_[1],l)))
            if s == 0 or s == float('nan') or s == float('-inf') or s == float('inf'):
                print("max is 0...Reshuffling")
                data = next(iter(loader))
                X = data["input_ids"].cpu().numpy()
                continue
            score = s
            if score == s:
                print(f"{iterations}: K = {K} score = {score}")
                #K+=2
            try:
                i = l[0][0]
            except:
                print("Random unidentified error...Reshuffling")
                data = next(iter(loader))
                X = data["input_ids"].cpu().numpy()
                continue
        return score, i, indices