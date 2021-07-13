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
        dataset, splits, _ = self._preprocessing(dataset, True)#, indices, Y_ 
        #indices.shuffle()
        init_epoch = self.classifier.curr_epoch
        training_args = TrainingArguments(output_dir='./results',
         num_train_epochs=self.classifier.final_epoch - init_epoch,
         do_train=True,
         per_device_train_batch_size=self.classifier.bs//4,
         per_device_eval_batch_size=self.classifier.bs//4,
         label_names=list(dataset.labels.keys()),
         label_smoothing_factor = 0.1,
         gradient_accumulation_steps=1,
         warmup_steps=500,
         weight_decay=1e-5,
         logging_dir='./logs',
         logging_strategy="steps",
         logging_steps=int((len(splits[0])//self.classifier.bs)*0.25),
         #evaluation_strategy="steps",
         #eval_steps=int((len(splits[1])//self.classifier.bs)*0.1),
         #eval_accumulation_steps = 1
         )
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
        loaders = [Loader(splits[0], self.classifier.bs, shuffle=True, num_workers=4),#, sampler=#indices[:splits[0]]),
        Loader(splits[1], self.classifier.bs, shuffle=True, num_workers=4),#, sampler=indices[splits[1]:]),
        Loader(splits[2], self.classifier.bs, shuffle=True, num_workers=4),#, sampler=indices[splits[1]+splits[2]:]),
        ]
        
        print("Dataset has been preprocessed and randomly split.\nRunning training loop...\n")
        metric = load_metric("accuracy")
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)
        
        """
        #print("\nRunning dimensoniality reduction...\nRunning training loop...\n")
        #self.classifier._k_means_approximation_one_step(loaders[0])
        """
        trainer = Trainer(model=self.classifier.model, args=training_args, train_dataset=splits[0], eval_dataset=splits[1], compute_metrics=compute_metrics)
        self.classifier.optimizer = trainer.optimizer
        self.classifier.scheduler = trainer.lr_scheduler
        while (self.classifier.curr_epoch < init_epoch + config["epochs"]):
            self.classifier.curr_epoch += 1
            print(f"Running epoch {self.classifier.curr_epoch}\n\n")
            trainer.model.train()
            print(trainer.prediction_loop(loaders[1],"batch training...", False,metric_key_prefix="training"))
            trainer.model.eval()
            trainer.optimizer.zero_grad()
            #print(trainer.evaluation_loop(loaders[0],description="Train split evaluation",prediction_loss_only=False))
            print(trainer.evaluation_loop(loaders[1],description="Validation split evaluation",prediction_loss_only=False))
            torch.cuda.empty_cache()
            if self.classifier.curr_epoch%config["save_interval"]==0:
                self.classifier._save(config["save_directory"], "{}-{}".format(self.classifier.name, self.classifier.curr_epoch))
            
        print("\n\n")
        print(trainer.evaluation_loop(loaders[0],description="Test split evaluation",prediction_loss_only=False))
        print("\nRun Complete.\n\n")

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
        indices = []
        #@TODO add features selection here
        if train:
            trainingValidationDatasetSize = int(0.6 * len(dataSetFolder))
            testDatasetSize = int(len(dataSetFolder) - trainingValidationDatasetSize) // 2
            #diff = len(dataSetFolder) - sum([trainingValidationDatasetSize, testDatasetSize, testDatasetSize+1])
            splits = [trainingValidationDatasetSize, testDatasetSize, testDatasetSize+1]
            splits = torch.utils.data.dataset.random_split(dataSetFolder, splits)
            #total = sum(list(dataSetFolder.distribution.values()))
            return dataSetFolder ,splits, indices#, Y
        return dataSetFolder