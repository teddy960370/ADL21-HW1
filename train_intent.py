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
from torch.utils.data import DataLoader
import numpy as np
from torchmetrics import Accuracy

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            300)
    model = model.to(device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #epoch_pbar = trange(args.num_epoch, desc="Epoch")
    epoch_pbar = trange(1, desc="Epoch")
    last_eval_acc = 0
    for epoch in epoch_pbar:
        
        acc = 0
        # TODO: Training loop - iterate over train dataloader and update model weights
        acc = train_model(train_loader,model,optimizer)
        print(f"Training Accuracy: {acc * 100} %")
        
        # TODO: Evaluation loop - calculate accuracy and save model weights
        acc = eval_model(test_loader,model)
        print(f"Eval Accuracy: {acc * 100} %")
        
        # early stop
        if(acc < last_eval_acc) :
            break
        
        last_eval_acc = acc
        pass

    # TODO: Inference on test set
    ckpt_path = args.ckpt_dir / "checkpoint.pt"
    torch.save(model.state_dict(), str(ckpt_path))

def train_model(data_loader, model, optimizer):
    loss_function = torch.nn.CrossEntropyLoss()
    num_batches = len(data_loader)
    model.train()
    total_acc = 0
    
    # model初始化
    model.zero_grad()
    
    for data in tqdm(data_loader):
        X = torch.from_numpy(np.array(data_loader.dataset.collate_fn('text',data['text']))).to(device)
        y = torch.from_numpy(np.array(data_loader.dataset.collate_fn('intent',data['intent']))).long()
        y = torch.nn.functional.one_hot(y,num_classes = 150).float().to(device)
        
        # 梯度清零
        optimizer.zero_grad()
        
        output = model(X)
        
        # 計算loss
        loss = loss_function(output,y)
        # 反向傳播
        loss.backward()
        # 更新參數
        optimizer.step()
        
        # 預測結果index
        pred = torch.argmax(output, dim=1)
        # 正解one hot index
        ans = torch.argmax(y, dim=1)

        accuracy = Accuracy().to(device)
        total_acc += accuracy(pred, ans)
    
    total_acc = total_acc / num_batches
    return total_acc
    #print(f"Training Accuracy: {total_acc * 100} %")
    
def eval_model(data_loader, model):

    num_batches = len(data_loader)
    total_acc = 0

    model.eval()
    with torch.no_grad():
        for data in tqdm(data_loader):
            X = torch.from_numpy(np.array(data_loader.dataset.collate_fn('text',data['text']))).to(device)
            y = torch.from_numpy(np.array(data_loader.dataset.collate_fn('intent',data['intent']))).long()
            y = torch.nn.functional.one_hot(y,num_classes = 150).float().to(device)
            output = model(X)

            pred = torch.argmax(output, dim=1)
            ans = torch.argmax(y, dim=1)
            
            accuracy = Accuracy().to(device)
            total_acc += accuracy(pred, ans)
        
    total_acc = total_acc / num_batches
    return total_acc
    #print(f"Eval Accuracy: {total_acc * 100} %")
    
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
