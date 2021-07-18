import numpy as np
import torchvision
import torch
import argparse
from experiment import Experiment
from torch.utils.data import DataLoader as Loader
from torchvision import transforms as transforms
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _model_config(args):
    config = {
        "model_name": args.model_name,
        "optimizer_name": args.optimizer,
        "criterion_name": args.loss,
        "dataset_directory":args.dataset_directory,
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "checkpoint": args.checkpoint if args.checkpoint else "",
        "num_classes": int(args.num_classes),
        "curr_epoch": int(args.curr_epoch) if args.curr_epoch else 0,
        "epochs": int(args.epochs) if args.epochs else 0,
        "train": True if args.train else False,
        "save_interval": int(args.save_interval),
        "library": args.library,
        "save_directory": args.save_directory,   
        "multi": True if args.multi else False,
        "drop": float(args.drop) if args.drop else 0,
        "log_step": int(args.log_step) if args.log_step else 1,
    }
    return config

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    np.seterr(divide='ignore', invalid='ignore')
    torch.backends.cudnn.enabled = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", "-m", help="Pick a model name")
    parser.add_argument("--dataset_directory", "-d", help="Set dataset directory path")
    parser.add_argument("--batch_size", "-b", help="Set batch size")
    parser.add_argument("--learning_rate", "-l", help="set initial learning rate")
    parser.add_argument("--checkpoint", "-c", help="Specify path for model to be loaded")
    parser.add_argument("--num_classes", "-n", help="set num classes")
    parser.add_argument("--curr_epoch", "-e", help="Set number of epochs already trained")
    parser.add_argument("--epochs", "-f", help="Train for these many more epochs")
    parser.add_argument("--optimizer", help="Choose an optimizer")
    parser.add_argument("--loss", help="Choose a loss criterion")
    parser.add_argument("--train", help="Set this model to train mode", action="store_true")
    parser.add_argument("--library")
    parser.add_argument("--save_directory", "-s")
    parser.add_argument("--save_interval", help="# of epochs to save checkpoints at.")
    parser.add_argument("--multi", help="Set this model to parallel mode", action="store_true")
    parser.add_argument("--drop")
    parser.add_argument("--log_step")


    args = parser.parse_args()
    config = _model_config(args)
    experiment = Experiment(config)
    if args.train:
        experiment._run()