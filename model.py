from math import log
from torch.autograd.functional import jacobian
from kmeans_pytorch import kmeans
import copy
import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
import numpy as np
import timm

class NLPClassifier(object):
    def __init__(self, config : dict):
        self.library = config["library"]
        self.nc = config["num_classes"]
        self.model, self.tokenizer = self._create_model(config["library"], config["model_name"], config["num_classes"])
        if config["train"]:
            self.optimizer = self._create_optimizer(config["optimizer_name"], self.model, config["learning_rate"])
            self.scheduler = self._create_scheduler(config["scheduler_name"], self.optimizer)
            self.criterion = self._create_criterion(config["criterion_name"])
        
        self.model = nn.DataParallel(self.model).cuda() if config["multi"] else self.model.cuda()
        #print(self.model)
        self.long = "long" in config["model_name"]
        if config["checkpoint"] != "":
            self._load(config["checkpoint"])
        self.curr_epoch = config["curr_epoch"]
        self.name = "{}-{}-{}".format(config["model_name"].split("/")[1] if "/" in config["model_name"] else config["model_name"], config["batch_size"], config["learning_rate"])
        self.bs = config["batch_size"]
        self.writer = SummaryWriter(log_dir="logs/{}".format(self.name))
        self.writer.flush()
        self.final_epoch = config["epochs"]
        self.best_cluster_center_score = float("-inf")
        self.score = float("-inf")
        print("Generated model: {}".format(self.name))

        
    def _create_model(self, library, model_name, num_classes):
        if library == "hugging-face":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if "roberta" in model_name:
                model.classifier.out_proj = nn.Linear(in_features=model.classifier.out_proj.in_features, out_features=num_classes, bias=True)
            elif not "long" in model_name: #TODO convert fine-tuned weights
                model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes, bias=True)
            else:
                model.classifier.out_proj = nn.Linear(in_features=model.classifier.out_proj.in_features, out_features=num_classes, bias=True)
            model.num_labels = num_classes
            return model, AutoTokenizer.from_pretrained(model_name)
        else:
            return timm.create_model(model_name, pretrained=True, num_classes=num_classes), AutoTokenizer.from_pretrained("bhadresh-savani/albert-base-v2-emotion")

    def _create_optimizer(self, name, model_params, lr):
        optim_dict = {"SGD":torch.optim.SGD(model_params.parameters(), lr, weight_decay=1e-5, momentum=0.9, nesterov=True),
                      "ADAM": torch.optim.Adam(model_params.parameters(), lr, betas=(0.9, 0.999)),
                      "ADAMW": torch.optim.AdamW(model_params.parameters(), lr, weight_decay=1e-5),
        }
        return optim_dict[name]
    
    def _create_scheduler(self, name, optimizer):
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR(optimizer, step_size=2.4, gamma=0.97),
            "CosineAnnealing": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 2)
        }
        return scheduler_dict[name]

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

    def _run_epoch(self, loaders):
        f1_train, acc_train, loss_train = self._train(loaders[0])
        f1_val, acc_val, loss_val = self._validate(loaders[1])
        self.curr_epoch += 1
        return f1_train, f1_val, acc_train, acc_val, loss_train, loss_val
    
    #TODO Abstract _train & _validate functions
    def _train(self, loader):
        self.model.train()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        #TODO self._k_means_approximation_one_step(loader) DO NOT REMOVE
        self._k_means_approximation_one_step(loader)
        #self.criterion.weight=torch.tensor([0 for _ in range(self.nc)]).cuda()
        #indices, k = self.clusters_idx, self.cluster_idx
        for data in loader:
            total += data["labels"].size(0)
            if self.library == "timm":
                shuffle_seed = torch.randperm(data["input_ids"].size(0))
                data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
                if self.score != float("-inf"):
                    x = data["input_ids"].view(data["input_ids"].size(0),3, -1)
                    x[:,:,self.clusters_idx!=self.cluster_idx] = 0
                    data["input_ids"] = x.view(x.size(0),3, 128, 128)
                outputs = self.model(data["input_ids"])
                loss = self.criterion(outputs, data["labels"])
            else:
                shuffle_seed = torch.randperm(data["input_ids"].size(0))
                data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
                if self.score != float("-inf"):
                    data["attention_mask"][:,self.clusters_idx!=self.cluster_idx] = 0
                outputs = self.model.forward(input_ids=data["input_ids"], attention_mask=data["attention_mask"]).logits
                #self.criterion.weight = torch.tensor([self.criterion.weight[i]+(data["labels"][data["labels"]==i].size(0)/self.bs) for i in range(16)]).cuda()
                loss = self.criterion(outputs.view(data["labels"].size(0), -1), data["labels"])
            self.optimizer.zero_grad()
            loss.backward()
            running_loss += loss.cpu().item()
            self.optimizer.step()
            self.scheduler.step()
            y_ = torch.argmax(outputs, dim=1)
            correct += (y_.cpu()==data["labels"].cpu()).sum().item()
            f1 += f1_score(data["labels"].cpu(), y_.cpu(), average='micro')
            iterations += 1
            torch.cuda.empty_cache()
            print(iterations, float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations))
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/total)


    def _validate(self, loader):
        self.model.eval()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        #indices, k = self.clusters_idx, self.cluster_idx
        #self._k_means_approximation_one_step(loader)
        with torch.no_grad():                
            for data in loader:
                if self.library == "timm":
                    shuffle_seed = torch.randperm(data["attention_mask"].size(0))
                    data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
                    if self.score != float("-inf"):
                        x = data["input_ids"].view(data["input_ids"].size(0),3, -1)
                        x[:,:,self.clusters_idx!=self.cluster_idx] = 0
                        data["input_ids"] = x.view(x.size(0),3, 128, 128)
                    outputs = self.model(data["input_ids"])
                    loss = self.criterion(outputs, data["labels"])
                else:
                    shuffle_seed = torch.randperm(data["input_ids"].size(0))
                    data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
                    if self.score != float("-inf"):
                        data["attention_mask"][:,self.clusters_idx] = 0
                    outputs = self.model.forward(input_ids=data["input_ids"], attention_mask=data["attention_mask"]).logits
                    loss = self.criterion(outputs.view(data["labels"].size(0), -1), data["labels"])
                running_loss += loss.cpu().item()
                y_ = torch.argmax(outputs, dim=1)
                correct += (y_.cpu()==data["labels"].cpu()).sum().item()
                f1 += f1_score(data["labels"].cpu(), y_.cpu(), average='micro')
                total += data["labels"].size(0)
                iterations += 1
                torch.cuda.empty_cache()
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)
    
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
        #f = copy.deepcopy(f)
        #f.zero_grad()
        if self.library == "timm":
            x = x["input_ids"].view(x["input_ids"].size(0),3, -1)
            x[:,:,clusters_idx!=cluster_idx] = 0
            x = x.view(x.size(0), x.size(1), 128,128)
            x.requires_grad = True
            preds = f(x)
            preds.backward(torch.ones_like(preds).cuda())
            J = x.grad
            #print(J.size())
            return J
        x["attention_mask"][:,clusters_idx!=cluster_idx] = 0
        x["attention_mask"].requires_grad = True
        y = x.pop("labels")
        preds = f(**x).logits
        preds.backward(torch.ones_like(preds).cuda())
        x["labels"] = y
        J = x["attention_mask"].grad
        x["attention_mask"].requires_grad = False
        x["attention_mask"][:,clusters_idx!=cluster_idx] = 1
        #print(J.size())
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