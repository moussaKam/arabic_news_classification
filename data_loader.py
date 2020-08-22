import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy

class NewsDataset(Dataset):
    def __init__(self, path_documents, path_labels, word2ind, max_len=512):
        self.max_len = max_len
        self.word2ind = word2ind
        self.documents = []
        self.labels = []
        with open(path_documents, 'r') as f1, open(path_labels, 'r') as f2:
            for line in f1:
                self.documents.append(line.strip())
            for line in f2:
                self.labels.append(int(line.strip()))
            assert len(self.documents) == len(self.labels)
                    
    def __len__(self):
        return len(self.documents)
        
    def __getitem__(self, index):
        sequence = self.documents[index].split()
        sequence = [self.word2ind[word] for word in sequence[:self.max_len]]
        sample = {'sequence': torch.tensor(sequence), 'label': self.labels[index]}
        return sample
    
class MyCollator(object):
    def __init__(self, device, pad_to_maxlen=False):
        """
        Args:
            device: cpu or cuda
            pad_to_maxlen: if True always pad to max length else pad to max sequence length within batch
        """
        self.device = device
        self.pad_to_maxlen = pad_to_maxlen
        
    def __call__(self, batch):
        if not self.pad_to_maxlen: # rnn
            sequences = pad_sequence([sample['sequence'] for sample in batch])
            labels = torch.tensor([sample['label'] for sample in batch],
                                    dtype=torch.long)
            return sequences.to(self.device), labels.to(self.device)
        else: #cnn
            pass
    
def get_loader(path_documents, path_labels, word2ind, device='cpu', max_len=512, batch_size=16):
        """
        Args:
            path_documents: path to file containing documents.
            path_labels: path to file containing labels
            word2ind: dictionnary mapping words to their respective index
            device: string: 'cpu' of 'cuda'
            batch_size: mini-batch size.
        Returns:
            data_loader: data loader for custom dataset.
        """
        collator = MyCollator(device)
        dataset = NewsDataset(path_documents, path_labels, word2ind, max_len=512)
        device=device.lower()
        assert (device=='cpu' or device=='cuda')
        data_loader = DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  collate_fn=collator,
                                                  drop_last=False,
                                                  )

        return data_loader
        
        