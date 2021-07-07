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
        print(self._features_selection(loaders[0]))

        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            f1_train, f1_val, train_acc, train_loss, val_acc, val_loss = self.classifier._run_epoch(loaders)
            print("Epoch: {} | Training Accuracy: {} | Training Loss: {} | Validation Accuracy: {} | Validation Loss: {} | f1 Train: {} | f1 Val  {}".format(self.classifier.curr_epoch, train_acc, train_loss, val_acc, val_loss, f1_train, f1_val))
            self.classifier.writer.add_scalar("Training Accuracy", train_acc, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Accuracy",val_acc, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Training Loss",train_loss, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("Validation Loss",val_loss, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Train",f1_train, self.classifier.curr_epoch)
            self.classifier.writer.add_scalar("f1 Val",f1_val, self.classifier.curr_epoch)
            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
        print("Testing:...")
        print(self.classifier._validate(loaders[2]))
        print("\nRun Complete.")

    def _preprocessing(self, directory, train):
        dataSetFolder = SpreadSheetNLPCustomDataset(directory, self.classifier.tokenizer)
        #@TODO add features selection here
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2           
            splits = torch.utils.data.random_split(dataSetFolder, [trainingValidationDatasetSize, testDatasetSize, testDatasetSize])
            split_names = ['train', 'validation', 'test']
            classes = list(dataSetFolder.labels.items())
            print(classes)
            distributions = {split_names[i]: {k: len(list(filter(lambda x: x["labels"]==v, splits[i]))) for k,v in classes} for i in range(len(splits))}
            print(distributions)
            return splits
        return dataSetFolder
    
    def _features_selection(self, loader):
        data = next(iter(loader))
        X = data["input_ids"].cpu().numpy()
        K = 2
        kmeans = KMeans(K)
        score = float("-inf")
        t_score = self.classifier._score(loader, indices=[i for i in range(X.shape[1])], -1)
        while t_score != score:
            indices = kmeans.fit_predict(X.T)
            z, indices = torch.tensor(X.T), torch.tensor(indices)
            clusters = [i for i in range(K)]
            t_score = score
            l = list(map(lambda idx: self.classifier._score(loader, indices, idx), clusters))
            i, score = torch.max(torch.tensor(l))
            K += 2
        return score, i, indices