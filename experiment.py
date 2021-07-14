from datasets import load_metric
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
from transformers import TrainingArguments
from transformers import Trainer

class Experiment(object):
    def __init__(self, config: dict):
        #TODO config.label2ids = 
        self.classifier = NLPClassifier(config)

    def _run(self, dataset, config: dict):
        _, splits, indices = self._preprocessing(dataset, True)#, indices, Y_ 
        #random.shuffle(indices)
        init_epoch = self.classifier.curr_epoch
        """
        training_args = TrainingArguments(output_dir='./results',
         label_names=list(self.classifier.dataset.labels.keys()),
         num_train_epochs=self.classifier.final_epoch - self.classifier.curr_epoch,
         gradient_accumulation_steps=1,
         per_device_train_batch_size=self.classifier.bs//4,
         warmup_steps=500,
         weight_decay=1e-5,
         do_train=True
         )
         """
        #weights.reverse()
        #self.classifier.criterion.weight = torch.tensor(weights).float().cuda()
        #random.shuffle(indices)
        #dist = [Y_[indices[:splits[0]]].size(0) for i in Y_.unique()]
        #beta = 0.999
        #effective_num = 1.0 - np.power(beta, dist)
        #weights = (1 - beta)/np.array(effective_num)
        #weights = weights/np.sum(weights)*self.classifier.nc
        #self.classifier.criterion = nn.CrossEntropyLoss(weight= torch.tensor(weights).float().cuda()).cuda()
        #splits = [dataset[:indices]]
        
        loaders = [Loader(splits[0], self.classifier.bs, shuffle=True, num_workers=4),
        Loader(splits[1], self.classifier.bs, shuffle=True, num_workers=4),
        Loader(splits[2], self.classifier.bs, shuffle=True, num_workers=4),
        ]
        
        print("Dataset has been preprocessed and randomly split.\nRunning training loop...\n")
        def compute_m(eval_pred):
            logits, labels = eval_pred
            y_ = np.argmax(logits, axis=-1)
            metric = load_metric("accuracy")
            return metric.compute(predictions=y_,references=labels)
        
        """
        #print("\nRunning dimensoniality reduction...\nRunning training loop...\n")
        #self.classifier._k_means_approximation_one_step(loaders[0])
        
        trainer = Trainer(model=self.classifier.model, args=training_args)
        self.classifier.optimizer = trainer.optimizer
        self.classifier.scheduler = trainer.lr_scheduler
        """
        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            self.classifier.curr_epoch +=1
            print(f"Train on entire seEpoch {self.classifier.curr_epoch}:\n\n")
            
            losses = []
            correct, total = 0,0
            self.classifier.model.train()
            for i, data in enumerate(loaders[0]):
                shuffle_seed = torch.randperm(data["input_ids"].size(0))
                data = {k:v[shuffle_seed].cuda() for k,v in list(data.items())}
                y = data.pop("labels")
                logits = self.classifier.model(**data).logits
                loss = self.classifier.criterion(logits.view(y.size(0), -1), y)
                losses.append(loss.mean().cpu().item())
                self.classifier.scaler.scale(loss).backward()
                self.classifier.optimizer.step()
                if i%2==0:
                    self.classifier.scheduler.step()
                    self.classifier.model.zero_grad()
                total += y.size(0)
                correct += (torch.argmax(logits, dim=-1).cpu()==y.cpu()).sum().item()
                print(i+1, sum(losses)/(i+1), correct/total)
            print("Training Metrics:", torch.mean(torch.tensor(losses)), correct/total," \n")
            
            losses = []
            correct, total = 0,0
            
            with torch.no_grad():
                self.classifier.model.eval()
                for i, data in enumerate(loaders[1]):
                    data = {k:v.cuda() for k,v in list(data.items())}
                    y = data.pop("labels")
                    logits = self.classifier.model(**data).logits
                    loss = self.classifier.criterion(logits.view(y.size(0), -1), y)
                    losses.append(loss.cpu().item())
                    total += y.size(0)
                    correct += (torch.argmax(logits, dim=-1).cpu()==y.cpu()).sum().item()
                    print(i+1, sum(losses)/(i+1), correct/total)
            print("Validation Metrics:", torch.mean(torch.tensor(losses)), correct/total," \n\n====================")
            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
            #trainer.state.num_train_epochs = 1
        
        print("\n\n")
        #print(trainer.evaluate(splits[2]))
        print("\nRun Complete.\n\n")

    def _preprocessing(self, directory, train):
        dataSetFolder = self.classifier.dataset       
        #loader = Loader(dataSetFolder, self.classifier.bs, shuffle=False, num_workers=4)
        
        print("\n\nRunning K-means for outlier detection...\n\n")
        """
        X = torch.tensor(torch.tensor(dataSetFolder.encodings["input_ids"])).cuda()
        X = X.view(X.size(0), -1)
        #Y = torch.tensor(np.asarray(dataSetFolder.dataset["type"].values))
        #Y = Y.view(Y.size(0), -1)
        #XY = torch.cat((X,Y), dim=1).cuda()
        cluster_ids_x, cluster_centers = kmeans(X=X, num_clusters=8, device=torch.device('cuda:0'))
        topk, indices = torch.topk(torch.tensor([(cluster_ids_x==i).nonzero().size(0) for i in range(8)]), 1)#torch.tensor([torch.mean(cluster_centers[i].float()).float() for i in range(8)]),2)#torch.tensor([(cluster_ids_x==i).nonzero().size(0) for i in range(8)]), 1)
        indices = torch.cat([(cluster_ids_x==i).nonzero() for i in indices], dim=0).view(-1).tolist()
        print(f"\n\nResult of k-means: {len(indices)} samples remain, taken from the top cluster(s)\n\n")
    
        #X_ = X[indices]
        #dist = [Y_[indices].cpu().size(0) for i in Y_.unique()]
        """
        #@TODO add features selection here
        #dataSetFolder = torch.utils.data.dataset.Subset(dataSetFolder,indices=indices)

        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2
            splits = [trainingValidationDatasetSize, testDatasetSize, testDatasetSize]
            diff = len(dataSetFolder) - sum(splits)
            #splits = torch.utils.data.dataset.random_split(dataSetFolder, [trainingValidationDatasetSize, testDatasetSize, testDatasetSize, diff])
            splits = [dataSetFolder, dataSetFolder, dataSetFolder]
            return dataSetFolder ,splits, []#, Y
        return dataSetFolder