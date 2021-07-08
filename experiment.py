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
        split, ds = self._preprocessing(dataset, True)
        init_epoch = self.classifier.curr_epoch
        loaders = [Loader(data, self.classifier.bs, shuffle=True, num_workers=4) for data in split]
        score, k, indices = self._features_selection(Loader(ds, self.classifier.bs, num_workers=4))
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
            return splits, dataSetFolder
        return dataSetFolder
    #@TODO...improve this...
    def _features_selection(self, loader):
        X = next(iter(loader))["input_ids"].cpu().numpy() #np.concatenate(tuple([data["input_ids"].cpu().numpy() for data in loader]), axis=0)
        K = 2
        score = float("-inf")
        i = -1
        t_score = [1e-4]
        Z = torch.tensor(X.T)
        iterations = 0
        while max(t_score) != score:
            iterations += 1
            X = next(iter(loader))["input_ids"].cpu().numpy()
            Z = torch.tensor(X.T)
            if max(t_score) < score:
                print(f"Updating...at {iterations}, done for {K} clusters, with score = {score}")
                K += K
                t_score.append(score)
            kmeans = KMeans(K, init="k-means++")
            indices = torch.tensor(kmeans.fit_predict(Z))
            clusters = {i: Z[indices==i] for i in range(K)}
            big_c = max(list(map(lambda c: len(c),list(clusters.values()))))
            #clusters = list(filter(lambda k: len(clusters[k])==big_c,))
            l = list(map(lambda idx: (idx, self.classifier._score(loader, indices, idx)), list(clusters.keys())))
            s = max(list(map(lambda l_: l_[1],l)))
            print(iterations, s)
            if float('nan') in list(map(lambda l_: l_[1],l)) or s == float('nan') or s == 0:
                K += K
                continue
            score = s
            l = list(filter(lambda a_: a_[1] == score, l))
            try:
                i = l[0][0]
            except:
                K += K
                continue
        return score, i, indices