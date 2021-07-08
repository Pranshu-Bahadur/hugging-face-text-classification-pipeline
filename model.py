import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForPreTraining, AutoConfig
from nfnets import SGD_AGC
from sam import SAMSGD
from sklearn.metrics import f1_score
import numpy as np

class NLPClassifier(object):
    def __init__(self, config : dict):
        self.model, self.tokenizer = self._create_model(config["library"], config["model_name"], config["num_classes"])
        if config["train"]:
            self.optimizer = self._create_optimizer(config["optimizer_name"], self.model, config["learning_rate"])
            self.scheduler = self._create_scheduler(config["scheduler_name"], self.optimizer)
            self.criterion = self._create_criterion(config["criterion_name"])
        self.model = nn.DataParallel(self.model).cuda()
        #print(self.model)
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
            model.classifier = nn.Linear(in_features=model.classifier.in_features, out_features=num_classes, bias=True)
            model.num_labels = num_classes
            return model, AutoTokenizer.from_pretrained(model_name)

    def _create_optimizer(self, name, model_params, lr):
        optim_dict = {"SGD":torch.optim.SGD(model_params.parameters(), lr),#,weight_decay=1e-5, momentum=0.9),#, nesterov=True
                      "ADAM": torch.optim.Adam(model_params.parameters(), lr, betas=(0.9, 0.999)),
                      "ADAMW": torch.optim.AdamW(model_params.parameters(), lr, betas=(0.9, 0.999)),
                      "SGDAGC": SGD_AGC(model_params.parameters(), lr=lr, clipping=0.08, weight_decay=1e-05, nesterov=True, momentum=0.9),
                      "SAMSGD": SAMSGD(model_params.parameters(), lr, momentum=0.9,weight_decay=1e-5)

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
        for batch in loader:
            """
            shuffle_seed = torch.randperm(batch["attention_mask"].size(0))
            batch = {k: v[shuffle_seed].cuda() for k, v in batch.items()}
            splits = [batch["labels"][batch["labels"]==y] for y in list(torch.unique(batch["labels"]))]
                #dist = [batch["labels"][batch["labels"]==y].size(0) for y in list(torch.unique(batch["labels"]))]
            bal = batch["labels"].size(0)//len(list(torch.unique(batch["labels"])))
            samples = [[i for i in range(bal)] if y.size(0) >= bal else [i for i in range(int(bal-y.size(0)))]+[i for i in range(y.size(0))] for y in splits]
            batch = {k: torch.stack([v[idx] for sample in samples for idx in sample]) for k,v in list(batch.items())}
            shuffle_seed = torch.randperm(batch["attention_mask"].size(0))
            print(batch["attention_mask"].size(0))
            """
            batch = {k: v[shuffle_seed].cuda() for k, v in batch.items()}
            batch["attention_mask"][:, indices!=k] = 0
            outputs = self.model.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = self.criterion(outputs.view(batch["input_ids"].size(0), self.nc), batch["labels"])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.scheduler.step()
            running_loss += loss.item()
            y_ = torch.argmax(outputs, dim=1)
            correct += (y_.cpu()==batch["labels"].cpu()).sum().item()
            f1 += f1_score(batch["labels"].cpu(), y_.cpu(), average='micro')
            total += batch["labels"].size(0)
            iterations += 1
            torch.cuda.empty_cache()
            print(iterations, float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations))
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)


    def _validate(self, loader, indices, k):
        self.model.eval()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        with torch.no_grad():                
            for batch in loader:
                shuffle_seed = torch.randperm(batch["attention_mask"].size(0))
                batch = {k: v[shuffle_seed].cuda() for k, v in batch.items()}
                batch["attention_mask"][:, indices!=k] = 0
                outputs = self.model.forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss = self.criterion(outputs.view(batch["input_ids"].size(0), -1), batch["labels"])
                running_loss += loss.item()
                y_ = torch.argmax(outputs, dim=1)
                correct += (y_.cpu()==batch["labels"].cpu()).sum().item()
                f1 += f1_score(batch["labels"].cpu(), y_.cpu(), average='micro')
                total += batch["labels"].size(0)
                iterations += 1
                torch.cuda.empty_cache()
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)
    #@TODO fix under-over sampling
    def _get_jacobian(self, data, indices, i):
        self.model.eval()
        self.model.zero_grad()
        """
        shuffle_seed = torch.randperm(data["attention_mask"].size(0))
        data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
        splits = [data["labels"][data["labels"]==y] for y in list(torch.unique(data["labels"]))]
        bal = data["labels"].size(0)//len(list(torch.unique(data["labels"])))
        samples = [[i for i in range(bal)] if y.size(0) >= bal else [i for i in range(bal-y.size(0))]+[i for i in range(y.size(0))] for y in splits]
        data = {k: torch.stack([v[idx] for sample in samples for idx in sample]) for k,v in list(data.items())}
        """
        shuffle_seed = torch.randperm(data["attention_mask"].size(0))
        data = {k: v[shuffle_seed].cuda() for k, v in data.items()}
        data["attention_mask"][:, indices!=i] = 0
        data["attention_mask"] = data["attention_mask"].float()
        data["attention_mask"].requires_grad = True
        h = self.model(data["input_ids"],attention_mask=data["attention_mask"]).logits.cuda()
        m = torch.zeros((data["attention_mask"].size(0), 16))
        print(data["attention_mask"].size(0))
        m[:,0] = 1
        h.backward(m.cuda())
        return data["attention_mask"].grad
    
    #@TODO Improve this...its nasty.
    def _score(self, data, indices, k):
        def eval_score_perclass(jacob, labels):
            if jacob is None or jacob.size(0) != labels.size(0):
                return 0
            try:
                K = 1e-3
                per_class={i.item(): jacob[labels==i].view(labels.size(0), -1) for i in list(torch.unique(labels))}
                ind_corr_matrix_score = {k: np.sum(np.log(np.absolute(np.corrcoef(v.cpu().numpy()+K))+K))+K/1e+2 for k,v in list(per_class.items())}
                score = np.sum(np.absolute(list(ind_corr_matrix_score.values())))
            except:
                return 0
            return score
        J = self._get_jacobian(data, indices, k)
        return eval_score_perclass(J, data['labels'].cuda())/1e+2


