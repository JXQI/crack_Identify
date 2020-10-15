from torch.utils.data import Dataset
import torch
from PIL import Image
from os.path import join

class dataloader(Dataset):
    def __init__(self,path,transforms=None,data_set='train'):
        self.path=path
        self.data_set=data_set
        self.data=path+'/'+data_set+'.txt'
        self.transform=transforms
        self.image=[]
        self.label=[]

        with open(self.data) as f:
            for line in f.readlines():
                line=line.strip().split()
                self.image.append(line[0])
                self.label.append(line[1])
    def __len__(self):
        return len(self.image)
    def __getitem__(self, item):
        image=Image.open(join(self.path,self.image[item]))
        label=torch.tensor(int(self.label[item]))
        if self.transform:
            image=self.transform(image)
        return image,label
if __name__=='__main__':
    d=dataloader('./data',data_set='test')
    print(len(d))


