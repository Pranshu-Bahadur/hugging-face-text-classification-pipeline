import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoConfig, AutoTokenizer, SqueezeBertForSequenceClassification
from nfnets import SGD_AGC
from sam import SAMSGD
from sklearn.metrics import f1_score
import numpy as np

class NLPClassifier(object):
    def __init__(self, config : dict):
        self.model, self.tokenizer = self._create_model(config["library"], config["model_name"], config['tokenizer_name'], config["num_classes"])
        if config["train"]:
            self.optimizer = self._create_optimizer(config["optimizer_name"], self.model, config["learning_rate"])
            self.scheduler = self._create_scheduler(config["scheduler_name"], self.optimizer)
            self.criterion = self._create_criterion(config["criterion_name"])
        self.model = nn.DataParallel(self.model).cuda()
        #self.tokenizer = self.tokenizer.cuda()
        if config["checkpoint"] != "":
            self._load(config["checkpoint"])
        self.curr_epoch = config["curr_epoch"]
        self.name = "{}-{}-{}".format(config["model_name"].split("/")[1], config["batch_size"], config["learning_rate"])
        self.bs = config["batch_size"]
        self.writer = SummaryWriter(log_dir="logs/{}".format(self.name))
        self.writer.flush()
        self.final_epoch = config["epochs"]
        self.nc = config["num_classes"]
        print("Generated model: {}".format(self.name))

        
    def _create_model(self, library, model_name, tokenizer, num_classes):
        if library == "hugging-face":
            return SqueezeBertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes), AutoTokenizer.from_pretrained(tokenizer)

    def _create_optimizer(self, name, model_params, lr):
        optim_dict = {"SGD":torch.optim.SGD(model_params.parameters(), lr,weight_decay=1e-5, momentum=0.9, nesterov=True),
                      "ADAM": torch.optim.Adam(model_params.parameters(), lr, betas=(0.9, 0.999)),
                      "SGDAGC": SGD_AGC(model_params.parameters(), lr=lr, clipping=0.01, weight_decay=1e-05, nesterov=True, momentum=0.9),
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
        return f1_train, f1_val, acc_train, acc_val, loss_train, loss_val

    def _train(self, loader, indices, k):
        self.model.train()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        for idx, batch in enumerate(loader):
            self.optimizer.zero_grad()
            x = batch['input_ids'].cuda()
            am = batch["attention_mask"].float().cuda()
            am[:, indices!=k] = 0
            y = batch['labels'].cuda()
            outputs = self.model.forward(x,  attention_mask=am).logits
            #loss = self.criterion(outputs, y)
            #loss.backward()
            if True:
                def closure():
                    self.optimizer.zero_grad()
                    preds = self.model(x.cuda())
                    loss = self.criterion(preds, y.cuda())
                    loss.backward()
                    return loss
                self.optimizer.zero_grad()
                loss = self.optimizer.step(closure)
            #self.optimizer.step()
            self.scheduler.step()
            running_loss += loss.item()
            y_ = torch.argmax(outputs, dim=1)
            correct += (y_.cpu()==y.cpu()).sum().item()
            f1 += f1_score(y.cpu(), y_.cpu(), average='micro')
            total += y.size(0)
            iterations += 1
            del x, y
            torch.cuda.empty_cache()
            print(idx, float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations))
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)


    def _validate(self, loader, indices, k):
        self.model.eval()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        with torch.no_grad():                
            for _, batch in enumerate(loader):
                x = batch['input_ids'].cuda()
                y = batch['labels'].cuda()
                am = batch["attention_mask"].float().cuda()
                #am[:, indices!=k] = 0
                outputs = self.model.forward(x, attention_mask=am).logits
                loss = self.criterion(outputs, y)
                running_loss += loss.item()
                y_ = torch.argmax(outputs, dim=1)
                correct += (y_.cpu()==y.cpu()).sum().item()
                f1 += f1_score(y.cpu(), y_.cpu(), average='micro')
                total += y.size(0)
                iterations += 1
                del x, y
                torch.cuda.empty_cache()
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)

    def _get_jacobian(self, loader, indices, i):
        data = next(iter(loader))
        x = data["attention_mask"].float().cuda()
        x.requires_grad = True
        h = self.model.forward(data["input_ids"][:,indices==i if i != -1 else indices].cuda(), attention_mask=x[:, indices==i if i != -1 else indices]).logits
        m = torch.zeros((x.size(0), self.nc))
        m[:, 0] = 1
        h.backward(m.cuda())
        return x.grad

    def _score(self, loader, indices, k):
        def eval_score_perclass(jacob, labels):
            K = 1e-5
            per_class={i.item(): jacob[labels==i].view(labels.size(0), -1) for i in list(torch.unique(labels))}
            ind_corr_matrix_score = {k: np.sum(np.log(np.absolute(np.corrcoef(v.cpu().numpy())+K))) for k,v in list(per_class.items())}
            return np.sum(np.absolute(list(ind_corr_matrix_score.values())))
        result = 0
        J = self._get_jacobian(loader, indices, k)
        for batch in loader:
            y = batch['labels'].cuda()
            try:
                result += eval_score_perclass(J, y)/1e+4
            except:
                continue
        return result/1e+4


