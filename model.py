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
            self.scheduler = self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 2.4, 0.97) #transformers.get_cosine_with_hard_restarts_schedule_with_warmup(self.optimizer, 500, self.final_epoch-self.curr_epoch)     # gamma factor --> 0.1
            self.criterion = self._create_criterion(config["criterion_name"])
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
    
    
    def run_epoch_step(self, loader, mode):
        total = 0
        metrics = ["accuracy","loss"]
        metrics = {f"{mode}-{metric}": [] for metric in metrics}
        self.model.train() # if mode =="train" else self.model.eval() # why did we comment model.eval()
        #if mode == "train":
            #self._k_means_approximation_one_step(loader)
        for i,data in enumerate(loader):
            #print(data)
            #inputs = data
            self.optimizer.zero_grad()
            #print('*'*3+'data size'+'*'*3+'\n')
            #print(str(data['labels'].size(0))+'\n')
            #shuffle_seed = torch.randperm(data.size(0))
            #print('*'*3+'shuffle seed'+'*'*3+'\n')
            #print(shuffle_seed)
            #x = {k:v[shuffle_seed].cuda() for k,v in list(data.items())}
            #if self.score != float("-inf") and mode == "train":
            #    x["attention_mask"][:,self.clusters_idx==self.cluster_idx] = 0
            #y = x.pop("labels")#x["labels"]##
            x = {k:v.cuda() for k,v in list(data.items())}
            y = x['labels']
            #x.pop('labels')
            #print("Labels")
            #print(y)
            total += y.size(0)
            #outputs = self.model(**x).logits
            #loss = self.criterion(outputs,labels)
            #outputs = self.model(**x)
            #loss, logits = outputs.loss.mean(), outputs.logits
            #logits = torch.nn.functional.dropout2d(outputs, self.drop) if mode == "train" else outputs
            #preds = F.softmax(outputs)
            #logits = torch.nn.functional.dropout2d(logits, self.drop) if mode == "train" else logits
            outputs = self.model(**x)
            logits = outputs.logits
            print(logits.size())
            logits = torch.nn.functional.dropout2d(logits, self.drop) if mode == "train" else logits
            loss = self.criterion(logits.view(logits.size(0), -1), y)             # use of reshaping the logits????
            #print("Predicted")
            #print(preds)
            #loss = self.criterion(preds, y)
            #print(loss)
            #preds = torch.argmax(preds,dim=1)
            print((torch.argmax(logits, dim=-1))
            metrics[f"{mode}-loss"].append(loss.cpu().item())
            metrics[f"{mode}-accuracy"].append((torch.argmax(logits, dim=-1).cpu()==y.cpu()).sum().item())
            if mode == "train": #TODO fix grad acc
                # loss.backward()
                self.scaler.scale(loss).backward() #TODO WTF does this even do?!
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()
                #self.log_step = int(len(loader)*0.1)
                if (i+1)%self.log_step==0:
                    print(f"Metrics at {i+1} iterations:\n",{k:sum(v)/(i+1) if "loss" in k else (sum(v)/total)*100 for k,v in list(metrics.items())}) #TODO naive logic used...
            del x, y
            torch.cuda.empty_cache()
        metrics = {k:sum(v)/len(loader) if "loss" in k else (sum(v)/total)*100 for k,v in list(metrics.items())}
        for k,v in list(metrics.items()):
            self.writer.add_scalar(k,v,self.curr_epoch)
        return metrics
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