import torch
from torch.utils.data import Dataset,DataLoader

class myDataset(Dataset):
    def __init__(self):
        self.data = torch.tensor([[1,2,3],[2,3,4],[3,4,5],[4,5,6]])
        self.target = torch.LongTensor([1,1,0,0])

    def __getitem__(self,index):
        return self.data[index],self.target[index]

    def __len__(self):
        return len(self.data)

mydataloader = DataLoader(dataset=myDataset(),
                          batch_size=1,
                          shuffle=False)

for i,(data,label) in enumerate(mydataloader):
    print(data,label)
