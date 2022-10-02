from typing import Dict

import torch
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        intput_size:int,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.intput_size = intput_size
        
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        
        self.lstm = torch.nn.LSTM(
            input_size = intput_size,
            hidden_size = hidden_size,
            batch_first = True,
            num_layers = num_layers,
            dropout = dropout,
            bidirectional = bidirectional
        )

        self.linear = torch.nn.Linear(in_features=hidden_size, out_features=1)
        self.seq = torch.nn.Sequential(
            
            torch.nn.Linear(1024, num_class),
            #torch.nn.ReLU(),
            #torch.nn.Softmax(),
            
            
            )

        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embedding = self.embed(batch)
        
        h0 = torch.zeros(self.num_layers, self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, self.hidden_size).requires_grad_()

        L, (hn, Cn) = self.lstm(embedding, (h0, c0))
        #out = self.linear(hn[0]).flatten()  # First dim of Hn is num_layers, which is set to 1 above.
        
        pred = self.linear(L).flatten() 
        result = self.seq(pred)
        return pred

        
        raise NotImplementedError


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
