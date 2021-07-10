from torch.autograd.functional import jacobian
from kmeans_pytorch import kmeans
import copy
import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForPreTraining, AutoConfig
#from nfnets import SGD_AGC
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
        self.best_cluster_center_score = 0
        self.score = 0
        print("Generated model: {}".format(self.name))

        
    def _create_model(self, library, model_name, num_classes):
        if library == "hugging-face":
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if not "long" in model_name:
                model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes, bias=True)
            else:
                model.classifier.out_proj = nn.Linear(in_features=model.classifier.out_proj.in_features, out_features=num_classes, bias=True)
            model.num_labels = num_classes
            return model, AutoTokenizer.from_pretrained(model_name)
        else:
            return timm.create_model(model_name, pretrained=True, num_classes=num_classes),  AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")

    def _create_optimizer(self, name, model_params, lr):
        optim_dict = {"SGD":torch.optim.SGD(model_params.parameters(), lr, weight_decay=1e-5, momentum=0.9, nesterov=True),#, nesterov=True),#
                      "ADAM": torch.optim.Adam(model_params.parameters(), lr, betas=(0.9, 0.999)),
                      "ADAMW": torch.optim.AdamW(model_params.parameters(), lr, betas=(0.9, 0.999),weight_decay=1e-5),
                      #"SGDAGC": SGD_AGC(model_params.parameters(), lr=lr, clipping=0.16, weight_decay=1e-05, nesterov=True, momentum=0.9),
        }
        return optim_dict[name]
    
    def _create_scheduler(self, name, optimizer):
        scheduler_dict = {
            "StepLR": torch.optim.lr_scheduler.StepLR(optimizer, step_size=2.4, gamma=0.97),
            "CosineAnnealing": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 2)
        }
        return scheduler_dict[name]

    def _create_criterion(self, name):
        loss_dict = {"CCE": nn.CrossEntropyLoss(weight=torch.tensor([0 for _ in range(self.nc)])).cuda(),
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

    def _train(self, loader):
        self.model.train()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        self._k_means_approximation_one_step(loader)
        indices, k = self.clusters_idx, self.cluster_idx
        for data in loader:
            
            """
            data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
            splits = [data["labels"][data["labels"]==y] for y in list(torch.unique(data["labels"]))]
                #dist = [data["labels"][data["labels"]==y].size(0) for y in list(torch.unique(data["labels"]))]
            bal = data["labels"].size(0)//len(list(torch.unique(data["labels"])))
            samples = [[i for i in range(bal)] if y.size(0) >= bal else [i for i in range(int(bal-y.size(0)))]+[i for i in range(y.size(0))] for y in splits]
            data = {k: torch.cat([v[idx] for sample in samples for idx in sample]) for k,v in list(data.items())}
            shuffle_seed = torch.randperm(data["attention_mask"].size(0))
            print(data["attention_mask"].size(0))
            """
            if self.library == "timm":
                shuffle_seed = torch.randperm(data["attention_mask"].size(0))
                data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
                #data["input_ids"].requires_grad = True
                #x = data["input_ids"].view(data["input_ids"].size(0),3, -1)
                #x[:,:,indices!=k] = 0
                #data["input_ids"] = x.view(self.bs, 3, 64, 64).float()
                outputs = self.model(data["input_ids"])
                loss = self.criterion(outputs, data["labels"])
            else:
                #data = self._splitter(data)
                #data["attention_mask"] = data["attention_mask"].view(-1, 4096)

                shuffle_seed = torch.randperm(data["input_ids"].size(0))
                data = {k: v[shuffle_seed].cuda() for k, v in data.items()}

                data["attention_mask"][:,indices!=k] = 0
                #data["attention_mask"] = data["attention_mask"].view(-1, 512)
                outputs = self.model.forward(input_ids=data["input_ids"]).logits
                #self.criterion.weight=torch.tensor([(data["labels"][data["labels"]==y].size(0)/self.bs) for y in range(self.nc)]).cuda()
                print(self.criterion.weight)
                loss = self.criterion(outputs.view(data["input_ids"].size(0), self.nc), data["labels"])

            #outputs = nn.functional.dropout2d(outputs, 0.2)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()
            y_ = torch.argmax(outputs, dim=1)
            correct += (y_.cpu()==data["labels"].cpu()).sum().item()
            f1 += f1_score(data["labels"].cpu(), y_.cpu(), average='micro')
            total += data["labels"].size(0)
            iterations += 1
            torch.cuda.empty_cache()
            print(iterations, float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations))
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)


    def _validate(self, loader):
        self.model.eval()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        indices, k = self.clusters_idx, self.cluster_idx
        with torch.no_grad():                
            for data in loader:
                if self.library == "timm":
                    shuffle_seed = torch.randperm(data["attention_mask"].size(0))
                    data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
                    #x = data["input_ids"].view(data["input_ids"].size(0),3, -1)
                    #x[:,:,indices!=k] = 0
                    #data["input_ids"] = x.view(self.bs, 3, 64, 64).float()
                    outputs = self.model(data["input_ids"])
                    loss = self.criterion(outputs, data["labels"])
                else:
                    #data = self._splitter(data)
                    #data["attention_mask"] = data["attention_mask"].view(-1, 4096)

                    shuffle_seed = torch.randperm(data["input_ids"].size(0))
                    data = {k: v[shuffle_seed].cuda() for k, v in data.items()}

                    data["attention_mask"][:,indices!=k] = 0
                    #data["attention_mask"] = data["attention_mask"].view(-1, 512)
                    outputs = self.model.forward(input_ids=data["input_ids"]).logits
                    loss = self.criterion(outputs.view(data["input_ids"].size(0), -1), data["labels"])
                running_loss += loss.item()
                y_ = torch.argmax(outputs, dim=1)
                correct += (y_.cpu()==data["labels"].cpu()).sum().item()
                f1 += f1_score(data["labels"].cpu(), y_.cpu(), average='micro')
                total += data["labels"].size(0)
                iterations += 1
                torch.cuda.empty_cache()
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)
    
    def _features_selection(self, K, loader, selection_heuristic=lambda x: torch.mode(x)):
        X = torch.cat([data["input_ids"] for data in loader][:-1])
        cluster_ids_x, cluster_centers = kmeans(X=X.T, num_clusters=2, device=torch.device('cuda:0'))
        best_cluster = selection_heuristic(cluster_ids_x)
        return best_cluster, cluster_centers[best_cluster], cluster_ids_x
    
    #From EPE-Nas (Note: Only for cases where num_classes < 100)
    #Given a Jacobian and target tensor calc epe-nase score.
    def _epe_nas_score_E(self, J_n, y_n):
        k = 1e-5
        V_J, V_y = (J_n - torch.mean(J_n)), (y_n - torch.mean(y_n))
        corr_m = torch.sum(V_J*V_y.T) / (torch.sqrt(torch.sum(V_J ** 2)) * torch.sqrt(torch.sum(V_y ** 2)))
        corr_m.apply_(lambda x: torch.log(abs(x)+k))
        return torch.sum(torch.abs(corr_m).view(-1)).item()
    
    #NOTE: Untested. Only Nlp
    #Given inputs X (dict of tensors of entire batch) return jacobian matrix on given function. 
    #Returns jacobian matrix for entire batch.
    def H_jacobian(self, f, X):
        f_sum = lambda x: torch.sum(f(X), axis=0)
        return jacobian(f_sum, X, vectorize=True).view(self.bs, -1)
    
    ##Given inputs X (dict of tensors of 1 batch) return jacobian matrix on given function.
    def _jacobian(self, f, x):
        x["attention_mask"][:,self.clusters_idx!=self.cluster_idx] = 0
        x["attention_mask"].requires_grad = True
        #y = x.pop("labels")
        #preds = f(**x).logits
        #preds.backward(torch.ones_like(preds))
        #x["labels"] = y
        J = jacobian(lambda x2: f(x["input_ids"], attention_mask=x2).logits, x["attention_mask"], vectorize=True).grad.detach()
        x["attention_mask"][:,self.clusters_idx!=self.cluster_idx] = 1
        return J
    
    def _epe_nas_score(self, loader):
        batches = [{k: v.float().cuda() if k == "attention_mask" else v.cuda() for k,v in list(data.items())}for data in loader]
        #NOTE: Possible error
        Y = torch.stack(list(map(lambda batch: batch["labels"], batches[:-1])))
        J = torch.stack(list(map(lambda batch: self._jacobian(self.model, batch).view(self.bs, -1),batches[:-1])))
        return self._epe_nas_score_E(J, Y)

    #@TODO Run intialization when model is created first.
    def _k_means_approximation_one_step(self, loader):
        best_cluster, best_cluster_center, clusters_idx = self._features_selection(2, loader)
        if torch.mean(best_cluster_center.view(-1)) > self.best_cluster_center_score:
            if self.score == 0:
                self.cluster_idx = best_cluster
                self.best_cluster_center = torch.sum(best_cluster_center.view(-1)) ##@?
                self.clusters_idx = clusters_idx
            score = self._epe_nas_score(loader)
            if score > self.score:
                self.cluster_idx = best_cluster
                self.best_cluster_center = torch.sum(best_cluster_center.view(-1)) ##@?
                self.clusters_idx = cluster_idx
                self.score = score


    
    





    

    




        

    
    #@TODO fix under-over sampling
    def _get_jacobian(self, data, indices, i):
        #self.model.eval()
        #self.model.zero_grad()
        #model = copy.deepcopy(self.model)
        #model = self.model
        #model.train()
        #model.zero_grad()
        """
        shuffle_seed = torch.randperm(data["attention_mask"].size(0))
        data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
        splits = [data["labels"][data["labels"]==y] for y in list(torch.unique(data["labels"]))]
        bal = data["labels"].size(0)//len(list(torch.unique(data["labels"])))
        samples = [[i for i in range(bal)] if y.size(0) >= bal else [i for i in range(bal-y.size(0))]+[i for i in range(y.size(0))] for y in splits]
        data = {k: torch.cat([v[idx] for sample in samples for idx in sample]) for k,v in list(data.items())}
        """
        if self.library != "timm":
            #data = self._splitter(data)

            #shuffle_seed = torch.randperm(data["attention_mask"].size(0))
            data = {k: v.cuda() for k, v in data.items()}

            #data["attention_mask"] = data["attention_mask"].view(-1, 4096)
            data["attention_mask"][:,indices!=i] = 0
            #data["attention_mask"] = data["attention_mask"].view(-1, 512)
            data["attention_mask"] = data["attention_mask"].float()
            data["attention_mask"].requires_grad = True
            #print(data["input_ids"], data["attention_mask"])
            #with torch.no_grad():
            h = self.model(data["input_ids"],attention_mask=data["attention_mask"]).logits
            #h.requires_grad = True
            m = torch.ones((self.bs, 16))
            #print(data["attention_mask"].size(0))
            #y_ = torch.argmax(h, dim=1)
            #m[:,y_] = 1
            #m[:,0] = 1
            h.backward(m.cuda())
            J = data["attention_mask"].grad
            #print(J.size())
            #J = J.view(data["attention_mask"].size(0), 1, 256, 256).float()
        else:
            #shuffle_seed = torch.randperm(data["attention_mask"].size(0))
            #data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
            x = data["input_ids"].view(data["input_ids"].size(0),3, -1)
            x[:,:,indices!=i] = 0
            data["input_ids"] = x.view(self.bs, 3, 64, 64).float()
            data["input_ids"].requires_grad = True
            h = self.model(data["input_ids"])
            m = torch.zeros((self.bs, 16))
            #print(data["attention_mask"].size(0))
            #y_ = torch.argmax(h, dim=1)
            #y_ = torch.argmax(h, dim=1)
            #m[:,y_] = 1
            m[:,0] = 1
            h.backward(m.cuda())
            J = data["input_ids"].grad

        return {"jacob": J, "data": data}
    
    #@TODO Improve this...its nasty.
    def _score(self, data, indices, k):
        def eval_score_perclass(jacob, data):
            labels = data["labels"]
            if torch.max(jacob).item() == 0 or jacob.size(0) != labels.size(0):
                return 0
            try:
                K = 1e-2
                score = np.sum(np.log(np.absolute(np.corrcoef(jacob.view(self.bs, -1).cpu().numpy(),labels.view(self.bs, -1).cpu().numpy()))))
            except:
                return 0
            print(score)
            return score
        #data = next(iter(loader))
        j_d = self._get_jacobian(data, indices, k)
        return eval_score_perclass(**j_d) #sum(list(map(lambda data:eval_score_perclass(j_d["J"], data["labels"])/1e+2, loader)))
    
    def _splitter(self, data):
        #[b.view(-1) for b in torch.tensor_split((data["input_ids"][data["labels"]==y]), 4096//512), dim=0)]
        splits = [(y, data["input_ids"][data["labels"]==y].view(-1,512), data["attention_mask"][data["labels"]==y].view(-1,512)) for y in list(torch.unique(data["labels"]))]
        splits = {split[0]: {
        "input_ids": split[1],
        "labels":torch.ones(split[1].size(0))*split[0],
        "attention_mask": split[2]} for split in splits}
        data = {k:torch.cat([split[k] for split in list(splits.values())]) for k in ["input_ids", "attention_mask", "labels"]}
        #print([v.size() for v in list(data.values())])
        shuffle_seed = torch.randperm(data["attention_mask"].size(0))
        data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
        return data