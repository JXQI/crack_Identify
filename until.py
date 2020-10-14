import torch
import matplotlib.pyplot as plt
# net:trained model
# dataloader:dataloader class
#loss_function: loss choose
#device: gpu or cpu
def Accuracy(net,dataloader,loss_function,device):
    loss_get=0
    total=0
    correct=0
    with torch.no_grad():
        for i,data in enumerate(dataloader,0):
            if i%100==99:
                inputs=data[0].to(device)
                labels=data[1].to(device)
                net=net.to(device)
                outputs=net(inputs)
                _,predicted=torch.max(outputs,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum().item()
                #print(predicted,labels)
                loss_get+=loss_function(outputs,labels)
        return loss_get/total,correct/total

def drawline(x,y,xlabel,ylabel,title):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.show()
