import json
import pickle
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier
from torch.utils.data import DataLoader,TensorDataset
import numpy as np

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
        
    # data loader
    train_loader = DataLoader(datasets['train'], batch_size = args.batch_size, shuffle = False)
    test_loader = DataLoader(datasets['eval'], batch_size = args.batch_size, shuffle = False)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
            embeddings,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.bidirectional,
            len(intent2idx),
            args.max_len)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_model(train_loader,model,optimizer)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        #eval_model(test_loader,model)
        pass

    # TODO: Inference on test set

def train_model(data_loader, model, optimizer):
    loss_function = torch.nn.CrossEntropyLoss()
    num_batches = len(data_loader)
    model.train()
    total_acc = 0
    for data in tqdm(data_loader):
        X = np.array(data['text'])
        y = np.array(data['intent'])
        output = model(X)
        loss = loss_function(output,y)
        loss.backward()

        optimizer.zero_grad()
        optimizer.step()
        
        #total_loss += loss.item()
        total_acc += (output.argmax(1) == y).sum().item()
    
    total_acc = total_acc / num_batches
    print(f"Accuracy: {total_acc}")

def eval_model(data_loader, model):

    num_batches = len(data_loader)
    total_acc = 0

    model.eval()
    with torch.no_grad():
        for X, y in tqdm(data_loader):
            X = X.cuda()
            y = y.cuda()
            output = model(X)
            total_acc += (output.argmax(1) == y).sum().item()

    total_acc = total_acc / num_batches
    print(f"Accuracy: {total_acc}")
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
