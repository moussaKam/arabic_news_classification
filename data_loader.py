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
            return len(documents)
        
        def __getitem__(self, index):
            sequence = self.documents.split()
            sequence = [word2ind[word] for word in sequence[:max_len]]
            sample = {'sequence': torch.tensor(sequence), 'label': labels[index]}
            return sample
        
    def collate_fn(data):
        sequences = pad_sequence([sample[i]['sequence'] for sample in data])
        labels = torch.tensor([sample['category'] for sample in data],
                             dtype=torch.long)
        return sequences, labels
    
    def get_loader(path_documents, path_labels, word2ind, device='cpu' max_len=512, batch_size=16):
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
        dataset = NewsDataset(path_documents, path_labels, path_dict, max_len=512)
        device=device.lower()
        assert (device=='cpu' or device=='cuda')
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  collate_fn=collate_fn,
                                                  drop_last=True,
                                                  pin_memory=(device=='cuda')
                                                  )

        return data_loader
        
        