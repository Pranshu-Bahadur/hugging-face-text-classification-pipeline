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
        self.nc = config["num_classes"]
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
        loss_dict = {"CCE": nn.CrossEntropyLoss().cuda(),
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

    def _run_epoch(self, loaders, indices, k):
        f1_train, acc_train, loss_train = self._train(loaders[0], indices, k)
        f1_val, acc_val, loss_val = self._validate(loaders[1], indices, k)
        self.curr_epoch += 1
        return f1_train, f1_val, acc_train, acc_val, loss_train, loss_val

    def _train(self, loader, indices, k):
        self.model.train()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
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
                data["input_ids"].requires_grad = True
                x = data["input_ids"].view(data["input_ids"].size(0), -1).clone()
                x[:,indices!=k] = 0
                data["input_ids"] = x.view(data["input_ids"].size()).float()
                outputs = self.model(data["input_ids"])
            else:
                #data = self._splitter(data)
                #data["attention_mask"] = data["attention_mask"].view(-1, 4096)

                shuffle_seed = torch.randperm(data["attention_mask"].size(0))
                data = {k: v[shuffle_seed].cuda() for k, v in data.items()}

                data["attention_mask"][:,indices!=k] = 0
                #data["attention_mask"] = data["attention_mask"].view(-1, 512)
                outputs = self.model.forward(input_ids=data["input_ids"], attention_mask=data["attention_mask"]).logits

            #outputs = nn.functional.dropout2d(outputs, 0.2)
            loss = self.criterion(outputs.view(data["input_ids"].size(0), self.nc), data["labels"])
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


    def _validate(self, loader, indices, k):
        self.model.eval()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        with torch.no_grad():                
            for data in loader:
                if self.library == "timm":
                    shuffle_seed = torch.randperm(data["attention_mask"].size(0))
                    data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
                    x = data["input_ids"].view(data["input_ids"].size(0), -1).clone()
                    x[:,indices!=k] = 0
                    data["input_ids"] = x.view(data["input_ids"].size()).float()
                    outputs = self.model(data["input_ids"])
                else:
                    #data = self._splitter(data)
                    #data["attention_mask"] = data["attention_mask"].view(-1, 4096)

                    shuffle_seed = torch.randperm(data["attention_mask"].size(0))
                    data = {k: v[shuffle_seed].cuda() for k, v in data.items()}

                    data["attention_mask"][:,indices!=k] = 0
                    #data["attention_mask"] = data["attention_mask"].view(-1, 512)
                    outputs = self.model.forward(input_ids=data["input_ids"], attention_mask=data["attention_mask"]).logits
                loss = self.criterion(outputs.view(data["input_ids"].size(0), -1), data["labels"])
                running_loss += loss.item()
                y_ = torch.argmax(outputs, dim=1)
                correct += (y_.cpu()==data["labels"].cpu()).sum().item()
                f1 += f1_score(data["labels"].cpu(), y_.cpu(), average='micro')
                total += data["labels"].size(0)
                iterations += 1
                torch.cuda.empty_cache()
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)
    #@TODO fix under-over sampling
    def _get_jacobian(self, data, indices, i):
        #self.model.eval()
        #self.model.zero_grad()
        model = copy.deepcopy(self.model)
        #model = self.model
        model.eval()
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

            shuffle_seed = torch.randperm(data["attention_mask"].size(0))
            data = {k: v[shuffle_seed].cuda() for k, v in data.items()}

            #data["attention_mask"] = data["attention_mask"].view(-1, 4096)
            data["attention_mask"][:,indices!=i] = 0
            #data["attention_mask"] = data["attention_mask"].view(-1, 512)
            data["attention_mask"] = data["attention_mask"].float()
            data["attention_mask"].requires_grad = True
            #print(data["input_ids"], data["attention_mask"])
            #with torch.no_grad():
            h = model(data["input_ids"],attention_mask=data["attention_mask"]).logits.cuda()
            #h.requires_grad = True
            m = torch.ones((data["input_ids"].size(0), 16))
            #print(data["attention_mask"].size(0))
            #m[:,0] = 1
            h.backward(m.cuda())
            J = data["attention_mask"].grad
            #print(J.size())
            #J = J.view(data["attention_mask"].size(0), 1, 256, 256).float()
        else:
            shuffle_seed = torch.randperm(data["attention_mask"].size(0))
            data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
            x = data["input_ids"].view(data["input_ids"].size(0), -1).clone()
            x[:,indices!=i] = 0
            data["input_ids"] = x.view(data["input_ids"].size()).float()
            data["input_ids"].requires_grad = True
            h = model(data["input_ids"])
            m = torch.ones((data["input_ids"].size(0), 16))
            #print(data["attention_mask"].size(0))
            #m[:,0] = 1
            h.backward(m.cuda())
            J = data["input_ids"].grad

        return {"jacob": J, "data": data}
    
    #@TODO Improve this...its nasty.
    def _score(self, loader, indices, k):
        def eval_score_perclass(jacob, data):
            labels = data["labels"].cuda()
            if jacob is None or jacob.size(0) != labels.size(0):
                return 0
            try:
                K = 1e-3
                score = sum(list(np.absolute(list(map(lambda i: np.sum(np.log(np.absolute(np.corrcoef(jacob[labels==i].view(labels.size(0), -1).cpu().numpy()+K))+K))+K/1e+2,list(torch.unique(labels)))))))
            except:
                return 0
            return score
        data = next(iter(loader))
        j_d = self._get_jacobian(data, indices, k)
        return eval_score_perclass(**j_d)/1e+2 #sum(list(map(lambda data:eval_score_perclass(j_d["J"], data["labels"])/1e+2, loader)))
    
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