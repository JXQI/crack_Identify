from model import Net
from Mul_models import models_select
from loader import dataloader
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from until import Accuracy,drawline
from os.path import join
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import operator
from functools import reduce

class Process:
    def __init__(self,device,model,batch_size,lr=0.1):
        self.device = device
        self.batch_size=batch_size
        self.lr=lr
        Net=models_select(class_num=2,pretrained=True)
        self.model=model
        self.net=Net.net(self.model)
        self.net=self.net.to(self.device)
        #self.transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        train_set=dataloader(path='./data',transforms=self.transform,data_set='train')
        val_set = dataloader(path='./data', transforms=self.transform,data_set='val')

        print(len(train_set),len(val_set))
        self.train_loader=DataLoader(dataset=train_set,batch_size=self.batch_size,shuffle=True,num_workers=0)
        self.val_loader = DataLoader(dataset=val_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.loss=nn.CrossEntropyLoss()
        self.optim=optim.SGD(self.net.parameters(),lr=self.lr,momentum=0.9,weight_decay=0.001)
        #存储结果最好的模型参数
        self.best_model = ''
    def train(self,epoch):
        loss_list=[]
        acc_list=[]
        max_acc=0
        self.best_model=''
        running_loss_arr = []
        for j in range(epoch):
            running_loss=0
            self.net.train()
            for i,data in enumerate(self.train_loader,0):
                self.optim.zero_grad()
                inputs,labels=data[0].to(self.device),data[1].to(self.device)
                print(inputs)
                output=self.net(inputs)
                #print(output,labels)
                loss=self.loss(output,labels)
                loss.backward() #计算梯度，反向传播
                self.optim.step()
                running_loss+=loss
                if i%100==99:
                    print("[%d, %d] loss:%f"%(j+1,i+1,running_loss/100))
                    #running_loss_arr.append(running_loss/100) #TODO:增加loss曲线的显示
                    running_loss=0
            loss_temp,acc_temp,loss_per=Accuracy(self.net,self.train_loader,self.loss,self.device)
            loss_list.append(loss_temp)
            acc_list.append(acc_temp)
            running_loss_arr.append(loss_per)
            print("%d epoch the loss is %f,the accuarcy is %f " %(j,loss_temp,acc_temp))
            #保存所有的model,并且挑出最好的
            model_name=self.model+'_'+str(j)+'_'+str(int(acc_temp*100))+'.pth'
            path='./Weights'
            if not os.path.isdir(path):
                os.mkdir(path)
            torch.save(self.net.state_dict(),join(path,model_name))
            if acc_temp>max_acc:
                max_acc=acc_temp
                self.best_model='best_'+model_name
        torch.save(self.net.state_dict(), join(path, self.best_model))
        drawline(range(epoch),loss_list,"epoch","loss","the loss of train")
        drawline(range(epoch),acc_list, "epoch","accuarcy", "the accuracy of train")
        running_loss_arr = reduce(operator.add, running_loss_arr)
        drawline(range(len(running_loss_arr)), running_loss_arr, "i", "loss", "the train_loss of the pre data") #TODO:增加loss的显示
        #plt.show()  #TODO:可以改造

    def validate(self):
        self.net.load_state_dict(torch.load(join('./Weights',self.best_model)))
        val_loss,val_acc,val_loss_arr=Accuracy(self.net,self.val_loader,self.loss,self.device)
        train_loss, train_acc, train_loss_arr = Accuracy(self.net, self.train_loader, self.loss, self.device)
        drawline(range(len(val_loss_arr)), val_loss_arr, "i", "loss", "the val_loss of the pre data")  # TODO:增加loss的显示
        drawline(range(len(train_loss_arr)), train_loss_arr, "i", "loss", "the train_best_loss of the pre data")  # TODO:增加loss的显示
        print("The vol_loss is %f ,The accuarcy is %f"%(val_loss,val_acc))
        print("The train_loss is %f ,The accuarcy is %f" % (train_loss, train_acc))

if __name__=="__main__":
    # device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)
    # pro=Process(device)
    # pro.train(epoch=2)
    # pro.validate()
    path='./Weights'
    if not os.path.isdir(path):
        os.mkdir(path)
