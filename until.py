import torch
import matplotlib.pyplot as plt
import os
# net:trained model
# dataloader:dataloader class
#loss_function: loss choose
#device: gpu or cpu
def Accuracy(net,dataloader,loss_function,device):
    loss_get=0
    loss=[]
    total=0
    correct=0
    net.eval()
    with torch.no_grad():
        for i,data in enumerate(dataloader,0):
            inputs=data[0].to(device)
            labels=data[1].to(device)
            net=net.to(device)
            outputs=net(inputs)
            _,predicted=torch.max(outputs,1)
            print(outputs, predicted,labels)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            #print(predicted,labels)
            temp=loss_function(outputs,labels)
            loss.append(temp)
            loss_get+=temp
        return loss_get/total,correct/total,loss

def drawline(x,y,xlabel,ylabel,title):
    path='./result'
    if not os.path.isdir(path):
        os.mkdir(path)
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.savefig('./result/'+title+'.jpg')
    #plt.show()  # TODO :为了同时显示多个图，将这个移除到最后，这里其实可以改为一个类的
