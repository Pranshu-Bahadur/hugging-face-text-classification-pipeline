from math import log
from torch.autograd.functional import jacobian
from kmeans_pytorch import kmeans
import copy
import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, PretrainedConfig#, AutoTokenizerFast
import numpy as np
from fairscale.optim.grad_scaler import ShardedGradScaler
from utils import SpreadSheetNLPCustomDataset
import transformers
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
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"], use_fast=True)
        self.dataset = SpreadSheetNLPCustomDataset(config['dataset_directory'], self.tokenizer)
        self.model_config = self._create_model_config(config["library"], config["model_name"], config["num_classes"], self.dataset.labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], config=self.model_config)##.from_config(config=self.model_config)#.
        self.model = nn.DataParallel(self.model).cuda() if config["multi"] else self.model.cuda()
        if config["train"]:
            self.optimizer = self._create_optimizer(config["optimizer_name"], self.model, config["learning_rate"])
            self.scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, 500, self.final_epoch-self.curr_epoch)
            self.criterion = self._create_criterion(config["criterion_name"])
        self.long = "long" in config["model_name"]
        if config["checkpoint"] != "":
            self._load(config["checkpoint"])
        self.name = "{}-{}-{}-{}-{}-{}".format(config["model_name"].split("/")[1] if "/" in config["model_name"] else config["model_name"], config["batch_size"], config["learning_rate"], config["optimizer_name"], "CosineAnnealing", config["criterion_name"])
        self.writer = SummaryWriter(log_dir="logs/{}".format(self.name))
        self.writer.flush()
        self.best_cluster_center_score = float("-inf")
        self.score = float("-inf")
        print("Generated model: {}".format(self.name))
        self.scaler = ShardedGradScaler() #if self.sharded_dpp else torch.cuda.amp.GradScaler(




    def _create_model_config(self, library, model_name, num_classes, labels_dict):
        if library == "hugging-face":
            config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
            config.id2label = {k:i for i,k in enumerate(labels_dict)}
            config.label2id = {str(i):k for i,k in enumerate(labels_dict)}
            return config

    def _create_optimizer(self, name, model_params, lr):
        optim_dict = {"SGD":torch.optim.SGD(model_params.parameters(), lr, weight_decay=1e-5, momentum=0.9, nesterov=True),
                      "ADAM": torch.optim.Adam(model_params.parameters(), lr, betas=(0.9, 0.999), eps=1e-8),
                      "ADAMW": torch.optim.AdamW(model_params.parameters(), lr,betas=(0.9, 0.999), weight_decay=1e-5, eps=1e-8, amsgrad=True),
        }
        return optim_dict[name]

    #TODO Add SOP loss
    def _create_criterion(self, name):
        loss_dict = {"CCE": nn.CrossEntropyLoss().cuda(),#weight=torch.tensor([0 for _ in range(self.nc)])).cuda(),
                     "MML": nn.MultiMarginLoss().cuda(),
                     "MSE": nn.MSELoss().cuda(),
                     "BCE": nn.BCELoss().cuda()
                     }
        return loss_dict[name]

    def _load(self, directory):
        print("loading previously trained model...")
        self.model.load_state_dict(torch.load(directory))

    def _save(self, directory, name):
        print("Saving trained {}...".format(name))
        torch.save(self.model.state_dict(), "{}/./{}.pth".format(directory, name))
    
    
    #TODO Abstract _train & _validate functions
    def run_epoch_step(self, loader, mode):
        total = 0
        metrics = ["accuracy","loss"]
        metrics = {f"{mode}-{metric}": [] for metric in metrics}
        self.model.train() if mode =="train" else self.model.eval() #TODO add with torch.no_grad()
        for i,data in enumerate(loader):
            x = {k:v.cuda() for k,v in list(data.items())}
            y = x.pop("labels")#x["labels"]
            total += y.size(0)
            logits = self.model(**x).logits
            #loss, logits = outputs.loss.mean(), outputs.logits TODO Not using model loss due to weighted loss computation.
            logits = torch.nn.functional.dropout2d(logits, 0.2) if mode == "train" else logits #TODO yeah i know...
            loss = self.criterion(logits.view(logits.size(0), -1), y)
            metrics[f"{mode}-loss"].append(loss.cpu().item())
            metrics[f"{mode}-accuracy"].append((torch.argmax(logits, dim=-1).cpu()==y.cpu()).sum().item())
            if mode == "train": #TODO fix grad acc
                self.scaler.scale(loss).backward() #TODO WTF does this even do?!
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                gradient_accumulation_steps = int(len(loader)*0.1)
                if (i+1)%gradient_accumulation_steps==0:
                    print(f"Metrics at {i+1} iterations:\n",{k:sum(v)/(i+1) if "loss" in k else (sum(v)/total)*100 for k,v in list(metrics.items())}) #TODO naive logic used...
        metrics = {k:sum(v)/len(loader) if "loss" in k else (sum(v)/total)*100 for k,v in list(metrics.items())}
        for k,v in list(metrics.items()):
            self.writer.add_scalar(k,v,self.curr_epoch)
        return metrics
    """
        Un-Implemented code (EPENAS/NAS-WOT) from this point....
    """
    #TODO Make sure this is using kmeans++
    def _features_selection(self, K, loader, selection_heuristic=lambda x: torch.mode(x)):
        X = torch.cat([data["input_ids"] for data in loader][:-1]).cuda()
        X = X.view(X.size(0), -1)
        cluster_ids_x, cluster_centers = kmeans(X=X.T, num_clusters=2, device=torch.device('cuda:0'))
        best_cluster, _ = selection_heuristic(cluster_ids_x)
        #print(best_cluster, cluster_centers[best_cluster], cluster_ids_x)
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
        preds.backward(torch.ones_like(preds).cuda())
        x["labels"] = y
        J = x["attention_mask"].grad
        x["attention_mask"].requires_grad = False
        x["attention_mask"][:,clusters_idx!=cluster_idx] = 1
        return J
    
    def _epe_nas_score(self, loader, clusters_idx, cluster_idx):
        batches = [{k: v.float().cuda() if k == "attention_mask" else v.cuda() for k,v in list(data.items())}for data in loader]
        Y = torch.tensor([]).cuda()
        J = torch.tensor([]).cuda()
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
        best_cluster, best_cluster_center, clusters_idx = self._features_selection(2, loader)
        print(best_cluster, torch.mean(best_cluster_center.view(-1)), clusters_idx)
        if torch.mean(best_cluster_center.view(-1)) > self.best_cluster_center_score:
            score = self._epe_nas_score(loader,clusters_idx, best_cluster)
            if score > self.score:
                self.cluster_idx = best_cluster
                self.best_cluster_center = torch.mean(best_cluster_center.view(-1)) ##@?
                self.clusters_idx = clusters_idx
                self.score = score
