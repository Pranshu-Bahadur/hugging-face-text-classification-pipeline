from math import log
from torch.autograd.functional import jacobian
from kmeans_pytorch import kmeans, pairwise_distance
import copy
import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig#, AutoTokenizerFast
import numpy as np
from fairscale.optim.grad_scaler import ShardedGradScaler
from utils import SpreadSheetNLPCustomDataset
import transformers
import uuid
from torch.utils.data import random_split as splitter
import torch.utils.data.dataloader as Loader
#from apex import amp

class NLPClassifier(object):
    def __init__(self, config : dict):
        self.library = config["library"]
        self.nc = config["num_classes"]
        self.curr_epoch = config["curr_epoch"]
        self.final_epoch = config["epochs"]
        self.bs = config["batch_size"]
        self.save_interval = config["save_interval"]
        self.save_directory = config["save_directory"]
        self.drop = config["drop"]
        self.log_step = config["log_step"]
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
        self.dataset = SpreadSheetNLPCustomDataset(config['dataset_directory'], self.tokenizer)
        self.model_config = self._create_model_config(config["library"], config["model_name"], config["num_classes"], self.dataset.labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], config=self.model_config, force_download=True)
        self.model = nn.DataParallel(self.model).to('cuda') if config["multi"] else self.model.to('cuda')    # figure out how to use distributed data parallel
        self.model.load_state_dict(self.model.state_dict())
        # print("State dict")
        # print(self.model.state_dict())
        # print(self.model.parameters())
        if config["train"]:
            self.optimizer = self._create_optimizer(config["optimizer_name"], self.model, config["learning_rate"])
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 2, 0.98) #transformers.get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, 500, self.final_epoch-self.curr_epoch)     # gamma factor --> 0.1
            self.criterion = self._create_criterion(config["criterion_name"])       # step size 2, gamma factor 0.98
        self.long = "long" in config["model_name"]                  # whats the point of long
        if config["checkpoint"] != "":
            self._load(config["checkpoint"])
        self.name = "{}-{}-{}-{}-{}-{}-{}".format(config["model_name"].split("/")[1] if "/" in config["model_name"] else config["model_name"],
         config["batch_size"], config["learning_rate"],
         config["optimizer_name"],
         "StepLR",
         config["criterion_name"],
         uuid.uuid4())
        self.writer = SummaryWriter(log_dir="logs/{}".format(self.name))
        self.writer.flush()
        self.best_cluster_center_score = float("-inf")
        self.score = float("-inf")
        print("Generated model: {}".format(self.name))
        self.scaler = ShardedGradScaler() #if self.sharded_dpp else torch.cuda.amp.GradScaler(
        self.best_acc = 0
        self.epochs_ran = 0
        self.best_weights = copy.deepcopy(self.model.state_dict())



    def _create_model_config(self, library, model_name, num_classes, labels_dict):
        if library == "hugging-face":
            config = AutoConfig.from_pretrained(model_name, num_labels=num_classes, force_download=True)
            config.id2label = {k:i for i,k in enumerate(labels_dict)}
            config.label2id = {str(i):k for i,k in enumerate(labels_dict)}
            return config

    def _create_optimizer(self, name, model_params, lr):
        optim_dict = {"SGD":torch.optim.SGD(model_params.parameters(), lr, weight_decay=1e-5, momentum=0.9, nesterov=True),
                      "ADAM": torch.optim.Adam(model_params.parameters(), lr, betas=(0.9, 0.999), eps=1e-8),
                      "ADAMW": torch.optim.AdamW(model_params.parameters(), lr,betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8),
        }
        return optim_dict[name]

    def _create_criterion(self, name):
        loss_dict = {"CCE": nn.CrossEntropyLoss().to('cuda'),
                     "MML": nn.MultiMarginLoss().to('cuda'),
                     "MSE": nn.MSELoss().to('cuda'),
                     "BCE": nn.BCELoss().to('cuda')
                     }
        return loss_dict[name]

    def _load(self, directory):
        print("loading previously trained model...")
        self.model.load_state_dict(torch.load(directory))

    def _save(self, directory, name):
        print("Saving trained {}...".format(name))
        torch.save(self.model.state_dict(), "{}/{}.pth".format(directory, name))
    
    
    #def run_epoch_step(self, loader, mode):
    def run_epoch_step(self, dataset, mode):
        print("number of epochs ran: ", str(self.epochs_ran))
        if mode == "train":
            self.epochs_ran += 1
        total = 0
        train_metrics = ["accuracy","loss"]
        train_metrics = {f"{metric}": [] for metric in train_metrics}

        test_metrics = ["accuracy","loss"]
        test_metrics = {f"{metric}": [] for metric in test_metrics}

        validation_metrics = ["accuracy","loss"]
        validation_metrics = {f"{metric}": [] for metric in validation_metrics}        # going with brute force for now

        # Running 4 fold 
        k_fold = 4
        print(len(dataset))
        
        subsets = splitter(dataset,[int(len(dataset)/k_fold)]*k_fold)           # generalize it later!


        self.model.train() # if mode =="train" else self.model.eval() # why did we comment model.eval()
        #print("Before decaying: "+str(self.scheduler.get_lr()))
        #print("After decaying: "+str(self.scheduler.get_lr()))
        #if mode == "train":
            #self._k_means_approximation_one_step(loader)
        if mode == "train":
            for k in range(k_fold):
                train_set=[]
                for j in subsets[:k]:           # using for loops now for time being, change it later!!
                    train_set.extend(j)
                for j in subsets[k+1:]:
                    train_set.extend(j)
                train_loader = Loader(train_set, self.bs, shuffle=True, num_workers=4)
                validation_set = subsets[k]
                validation_loader = Loader(validation_set, self.bs, shuffle=False, num_workers=4)           # might have to change to true
                for i,data in enumerate(train_loader):
                    x = {key:v.cuda() for key,v in list(data.items())}
                    y = x['labels']
                    total += y.size(0)
                    outputs = self.model(**x)
                    logits = outputs.logits
                    logits = torch.nn.functional.dropout2d(logits, 0.15) if mode == "train" else logits       # dropout prob at 0.15
                    loss = self.criterion(logits.view(logits.size(0), -1), y)             # use of reshaping the logits????   gives error in 100's 
                    train_metrics["loss"].append(loss.cpu().item())
                    train_metrics["accuracy"].append((torch.argmax(logits, dim=-1).cpu()==y.cpu()).sum().item())
                    if mode == "train": #TODO fix grad acc
                        # loss.backward()
                        self.scaler.scale(loss).backward() #TODO WTF does this even do?!
                        self.optimizer.step()
                        self.scheduler.step()          # decay weight every time in a mini batch?
                        self.model.zero_grad()
                        #self.log_step = int(len(loader)*0.1)
                        #if (i+1)%self.log_step==0:
                            #print(f"Metrics at {i+1} iterations:\n",{k:sum(v)/(i+1) if "loss" in k else (sum(v)/total)*100 for k,v in list(train_metrics.items())}) #TODO naive logic used...
                    del x, y
                    torch.cuda.empty_cache()
                # permutation training accuracy
                train_metrics = {k:sum(v)/len(train_loader) if "loss" in k else (sum(v)/total)*100 for k,v in list(train_metrics.items())}
                for i,data in enumerate(validation_loader):
                    mode = "validation"

                    x = {key:v.cuda() for key,v in list(data.items())}
                    y = x['labels']
                    total += y.size(0)
                    outputs = self.model(**x)
                    logits = outputs.logits
                    loss = self.criterion(logits.view(logits.size(0), -1), y)             # use of reshaping the logits????   gives error in 100's 
                    validation_metrics["loss"].append(loss.cpu().item())
                    validation_metrics["accuracy"].append((torch.argmax(logits, dim=-1).cpu()==y.cpu()).sum().item())
                    del x, y
                    torch.cuda.empty_cache()
                validation_metrics = {k:sum(v)/len(validation_loader) if "loss" in k else (sum(v)/total)*100 for k,v in list(validation_metrics.items())}
            train_metrics = {k:sum(v)/k_fold if "loss" in k else (sum(v)/k_fold)*100 for k,v in list(train_metrics.items())}    # calculating for an epoch
            validation_metrics = {k:sum(v)/k_fold if "loss" in k else (sum(v)/k_fold)*100 for k,v in list(validation_metrics.items())}
            print("train metrics:",str(train_metrics))
            print("Validation metrics:",str(validation_metrics))
            return (train_metrics,validation_metrics)
        if mode == "test":
            for i,data in enumerate(dataset):           # function is called with loaded test data
                    x = {key:v.cuda() for key,v in list(data.items())}
                    y = x['labels']
                    total += y.size(0)
                    outputs = self.model(**x)
                    logits = outputs.logits
                    loss = self.criterion(logits.view(logits.size(0), -1), y)             # use of reshaping the logits????   gives error in 100's 
                    test_metrics[f"{mode}-loss"].append(loss.cpu().item())
                    test_metrics[f"{mode}-accuracy"].append((torch.argmax(logits, dim=-1).cpu()==y.cpu()).sum().item())
                    del x, y
                    torch.cuda.empty_cache()
            test_metrics = {k:sum(v)/k_fold if "loss" in k else (sum(v)/k_fold)*100 for k,v in list(test_metrics.items())}
            print(test_metrics)
            return test_metrics
        #metrics = {k:sum(v)/len(loader) if "loss" in k else (sum(v)/total)*100 for k,v in list(metrics.items())}
        # train_metrics = {k:sum(v)/k_fold if "loss" in k else (sum(v)/k_fold)*100 for k,v in list(train_metrics.items())}
        # validation_metrics = {k:sum(v)/k_fold if "loss" in k else (sum(v)/k_fold)*100 for k,v in list(validation_metrics.items())}
        # test_metrics = {k:sum(v)/k_fold if "loss" in k else (sum(v)/k_fold)*100 for k,v in list(test_metrics.items())}
        # print(train_metrics)
        # print(validation_metrics)
        # curr_acc = list(metrics.items())[0][1]
        # for k,v in list(metrics.items()):
        #     self.writer.add_scalar(k,v,self.curr_epoch)
        # if curr_acc > self.best_acc:
        #     self.best_acc, self.best_weights = curr_acc, copy.deepcopy(self.model.state_dict())
        # if self.best_acc < curr_acc and mode == "train" and (self.epochs_ran)%2 == 0:       # what if i just do it while training
        #     self.best_acc = curr_acc
        #     self.model.load_state_dict(self.best_weights)
        # #self.scheduler.step()               # decaying weight once per epoch is not enough? i was just experimenting lol
        # return metrics
    """
        Un-Implemented code (EPENAS/NAS-WOT) from this point....
    """
    #TODO Make sure this is using kmeans++ DEPRECEATED
    def _features_selection_(self, K, loader, selection_heuristic=lambda x: torch.mode(x)):
        X = torch.cat([data["input_ids"] for data in loader][:-1]).to('cuda')
        X = X.view(X.size(0), -1)
        cluster_ids_x, cluster_centers = kmeans(X=X.T, num_clusters=2, device=torch.device('cuda:0'))
        best_cluster, _ = selection_heuristic(cluster_ids_x)
        #print(best_cluster, cluster_centers[best_cluster], cluster_ids_x)
        return best_cluster, cluster_centers[best_cluster], cluster_ids_x
    
    #TODO add topk here as well.
    def _features_selection(self, loader, n, selection_heuristic=lambda x: torch.mode(x)):
        X = torch.cat([data["input_ids"] for data in loader][:-1]).to('cuda')
        X = X.view(X.size(0), -1)
        m_dict = {}
        differences = [0]
        for k in range(2, n+1):
            cluster_ids, centers  = kmeans(X=X.T, num_clusters = k, device=torch.device('cuda'))
            curr_inertia = sum([torch.sum((1/(2*i+1))*pairwise_distance(X.T[cluster_ids==i], centers[i]), 0).cpu().item() for i in range(k)])/1e+5
            if k!=2:
                prev_inertias = list(m_dict.keys())
                difference = int(sum(prev_inertias)/len(prev_inertias)) - int((sum(prev_inertias) - curr_inertia)/len(prev_inertias)+1)
                if len(differences)>2 and differences[-1] < difference: #abs(max(differences) - difference)
                    print(f"{self.curr_epoch} Elbow at {k-1}")
                    break
                differences.append(difference)
            m_dict[curr_inertia] = {"k": k, "cluster_ids": cluster_ids, "centers": centers}
        result = m_dict[list(m_dict.keys())[-1]]
        k, cluster_ids_x, cluster_centers = result["k"], result["cluster_ids"], result["centers"]
        best_cluster = selection_heuristic(cluster_ids_x)
        return best_cluster, cluster_centers[best_cluster], cluster_ids_x

    
    #From EPE-Nas (Note: Only for cases where num_classes < 100)
    #Given a Jacobian and target tensor calc epe-nase score.
    #TODO Add vectorized classwise correlation...currently like NAS-WOT
    def _epe_nas_score_E(self, J_n, y_n):
        k = 1e-5
        V_J, V_y = (J_n - torch.mean(J_n)), (y_n - torch.mean(y_n))
        #print(V_J.size(), V_y.size())
        corr_m = torch.sum(V_J.T*V_y) / (torch.sqrt(torch.sum(V_J.T ** 2)) * torch.sqrt(torch.sum(V_y ** 2)))
        corr_m = torch.log(torch.abs(corr_m)+k)
        return torch.sum(torch.abs(corr_m).view(-1)).item()
    
    ##Given inputs X (dict of tensors of 1 batch) return jacobian matrix on given function.
    def _jacobian(self, f, x, clusters_idx, cluster_idx):
        x["attention_mask"][:,clusters_idx!=cluster_idx] = 0
        x["attention_mask"].requires_grad = True
        y = x.pop("labels")
        preds = f(**x).logits
        preds.backward(torch.ones_like(preds).to('cuda'))
        x["labels"] = y
        J = x["attention_mask"].grad
        x["attention_mask"].requires_grad = False
        x["attention_mask"][:,clusters_idx!=cluster_idx] = 1
        return J
    
    def _epe_nas_score(self, loader, clusters_idx, cluster_idx):
        batches = [{k: v.float().to('cuda') if k == "attention_mask" else v.to('cuda') for k,v in list(data.items())} for data in loader]
        Y = torch.tensor([]).to('cuda')
        J = torch.tensor([]).to('cuda')
        iterations = 0
        score = 0
        for batch in batches:
            iterations+=1
            J_ = self._jacobian(self.model, batch, clusters_idx, cluster_idx)
            J = torch.cat([J, J_.view(J_.size(0), -1)])
            Y = torch.cat([Y,batch["labels"]]).float()
            score += self._epe_nas_score_E(J, Y)
            print(f"{iterations}: accumluated score = {score}")
            if score/len(batch)*self.bs > self.score/len(batch)*self.bs:
                print(f"Score at {iterations}: {score}. Is better than mean of previous best score. Pruning")
                return score
        return score

    #@TODO Run intialization when model is created first.
    def _k_means_approximation_one_step(self, loader):
        best_cluster, best_cluster_center, clusters_idx = self._features_selection(loader, 256)
        if torch.mean(best_cluster_center.view(-1)) > self.best_cluster_center_score:
            score = self._epe_nas_score(loader,clusters_idx, best_cluster)
            if score > self.score:
                self.cluster_idx = best_cluster
                self.best_cluster_center = torch.mean(best_cluster_center.view(-1)) ##@?
                self.clusters_idx = clusters_idx
                self.score = score