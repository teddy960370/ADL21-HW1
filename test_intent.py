import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    
    test_loader = DataLoader(dataset, batch_size = 1, shuffle = False)


    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        300
    )
    model.to(device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    
    # load weights into model
    model.load_state_dict(ckpt)
    
    # TODO: predict dataset
    ans = test_model(test_loader,model)
    
    # TODO: write prediction to file (args.pred_file)
    data = pd.DataFrame()
    for index,(answer) in enumerate(ans) :
        d = {'id':'test-' + str(index),'intent':answer}
        data = data.append(d,ignore_index=True)
        
    data.to_csv(args.pred_file,index=False)
        
def test_model(data_loader, model):
    ans = list()
    with torch.no_grad():
        for data in tqdm(data_loader):
            X = torch.from_numpy(np.array(data_loader.dataset.collate_fn('text',data['text']))).to(device)

            output = model(X)
            pred = torch.argmax(output, dim=1)
            pred = data_loader.dataset.idx2label(pred.item())
            ans.append(pred)

    return ans
    #print(f"Eval Accuracy: {total_acc * 100} %")

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        default="./data/intent/test.json",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        default="./ckpt/intent/checkpoint.pt",
    )
    parser.add_argument("--pred_file", type=Path, default="./data/intent/intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
