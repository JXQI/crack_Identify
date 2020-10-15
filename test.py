import matplotlib.pyplot as plt
import numpy as np
import os
from Mul_models import models_select
import torch
from PIL import Image
from os.path import join
from torchvision import transforms
import torchvision

def imgshow(img):
    img=img/2+0.5
    npimg=img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

class Test:
    def __init__(self,model,weight_path=None,image_path='./samples',transforms=None):
        self.weight_path = weight_path
        self.image_path=image_path
        self.transforms=transforms
        self.model=model
        Net = models_select(class_num=2, pretrained=True)
        self.net = Net.net(self.model)
        #self.net.load_state_dict(torch.load(self.weight_path))
    def result(self):
        for image in os.listdir(self.image_path):
            img=Image.open(join(self.image_path,image))
            img=self.transforms(img)
            imgshow(img)
            img = img.unsqueeze(0)
            output=self.net(img)
            print(output)
            _, predicted = torch.max(output, 1)
            print(image,predicted)

if __name__=='__main__':
    if not os.path.isdir('./samples'):
        os.mkdir('./samples')
    #transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
    #                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    T=Test(model='ResNet50',transforms=transform)
    T.result()
