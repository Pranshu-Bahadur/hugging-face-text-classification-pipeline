import torch
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoTokenizer
from sklearn.metrics import f1_score

class NLPClassifier(object):
    def __init__(self, config : dict):
        self.model, self.tokenizer = self._create_model(config["library"], config["model_name"], config['tokenizer'], config["num_classes"])
        if config["train"]:
            self.optimizer = self._create_optimizer(config["optimizer_name"], self.model, config["learning_rate"])
            self.scheduler = self._create_scheduler(config["scheduler_name"], self.optimizer)
            self.criterion = self._create_criterion(config["criterion_name"])
        self.model = nn.DataParallel(self.model).cuda()
        self.tokenizer = nn.DataParallel(self.tokenizer).cuda()
        if config["checkpoint"] != "":
            self._load(config["checkpoint"])
        self.curr_epoch = config["curr_epoch"]
        self.name = "{}-{}-{}-{}-{}".format(config["model_name"], config["tokenizer"], config["resolution"], config["batch_size"], config["learning_rate"])
        self.bs = config["batch_size"]
        self.writer = SummaryWriter(log_dir="logs/{}".format(self.name))
        self.writer.flush()
        self.final_epoch = config["epochs"]
        self.nc = config["num_classes"]
        print("Generated model: {}".format(self.name))

        
    def _create_model(self, library, model_name, tokenizer, num_classes):
        if library == "hugging-face":
            return AutoConfig.from_pretrained(model_name, num_labels=num_classes), AutoTokenizer.from_pretrained(tokenizer)

    def _create_optimizer(self, name, model_params, lr):
        optim_dict = {"SGD":torch.optim.SGD(model_params.parameters(), lr,weight_decay=1e-5, momentum=0.9, nesterov=True),
                      "ADAM": torch.optim.Adam(model_params.parameters(), lr, betas=(0.9, 0.999))
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

    def _run_epoch(self, split):
        f1_train, acc_train, loss_train = self._train(split[0])
        f1_val, acc_val, loss_val = self._train(split[0])
        return f1_train, f1_val, acc_train, acc_val, loss_train, loss_val

    def _train(self, loader):
        self.model.train()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        for batch in loader:
            self.optimizer.zero_grad()
            x_ids = batch['input_ids'].cuda()
            x = batch['attention_mask'].cuda()
            y = batch['labels'].cuda()
            outputs = self.model(x_ids,attention_mask=x, labels=y)[1]
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            training_loss += loss.item()
            y_ = torch.argmax(outputs, dim=1)
            correct += (y_.cpu()==y.cpu()).sum().item()
            f1 += f1_score(y.cpu(), y_.cpu(), average='micro')
            total += y.size(0)
            iterations += 1
            del x, y
            torch.cuda.empty_cache()
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)


    def _validate(self, loader):
        self.model.eval()
        running_loss, correct, iterations, total, f1 = 0, 0, 0, 0, 0
        with torch.no_grad():                
            for batch in loader:
                self.optimizer.zero_grad()
                x_ids = batch['input_ids'].cuda()
                x = batch['attention_mask'].cuda()
                y = batch['labels'].cuda()
                outputs = self.model(x_ids,attention_mask=x, labels=y)[1]
                loss = self.criterion(outputs, y)
                training_loss += loss.item()
                y_ = torch.argmax(outputs, dim=1)
                correct += (y_.cpu()==y.cpu()).sum().item()
                f1 += f1_score(y.cpu(), y_.cpu(), average='micro')
                total += y.size(0)
                iterations += 1
                del x, y
                torch.cuda.empty_cache()
        return float(f1/float(iterations))*100, float(correct/float(total))*100, float(running_loss/iterations)

    def _get_jacobian(self, loader):
        data = next(iter(loader))
        x = data["input_ids"].cuda()
        self.model.zero_grad()
        J = torch.autograd.functional.jacobian(lambda x_: self.model(x_.long())[0], x.float()).cuda()
        return J


