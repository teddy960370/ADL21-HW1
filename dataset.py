from typing import List, Dict

from torch.utils.data import Dataset

from utils import Vocab


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

        #self.collate_fn()
    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self,inputType: str, Myinput : List[str]):
        # TODO: implement collate_fn
        #x,y = {x: intent,y: intent for text, intent in List}
        if(inputType == 'text') :
            text = self.vocab.encode_batch(Myinput,128)
            return text
        else :
            intent = [self.label2idx(data) for data in Myinput]
            return intent
        
        #return text,intent

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len
    
        #self.collate_fn()
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance


    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self,inputType: str, Myinput : List[str]):
        # TODO: implement collate_fn
        #x,y = {x: intent,y: intent for text, intent in List}
        if(inputType == 'text') :
            text = self.vocab.encode_batch(Myinput,128)
            return text
        else :
            intent = [self.label2idx(data) for data in Myinput]
            return intent
        
        #return text,intent

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]
        
        
