import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=nn.Conv2d(3,6,5) #pannel,kernel数目,kernel大小
        self.conv2=nn.Conv2d(6,16,5)
        self.pool=nn.MaxPool2d(2,2)
        self.fc1=nn.Linear(16*53*53,30)
        self.fc2=nn.Linear(30,11)
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x=x.view(-1,16*53*53)   #这里是将所有的特征flatten,是需要计算的，而不是随便给的
        x=F.relu(self.fc1(x))
        #x=F.relu(self.fc2(x))  #TODO:注意最后一层不能加激活函数
        x=self.fc2(x)

        return x
#
class Alex_Net(nn.Module):
    def __init__(self):
        super(Alex_Net,self).__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            # TODO:LRN
            nn.Conv2d(96,256,kernel_size=5,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #TODO LRN
            nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256,kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.classifier=nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=256*6*6,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(inplace=True),
            #nn.Dropout(),  #为何这里没有Dropout，因为这里只是一个简单的映射，不能丢失特征信息
            nn.Linear(in_features=4096,out_features=1000),
        )

    def forward(self,x):
            x=self.features(x)
            x = torch.flatten(x, 1)
            x=self.classifier(x)
            return  x
class VGG_Net(nn.Module):
    def __init__(self):
        super(VGG_Net,self).__init__()
        self.features=nn.Sequential(
            #block1
            nn.Conv2d(3,64,3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1,stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #block2
            nn.Conv2d(64,128,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
            #block3
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #block4
            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # block5
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier=nn.Sequential(
            #nn.Dropout(), #TODO:为何这里没有Dropout
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096,out_features=4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_features=4096,out_features=1000),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7)) #TODO:何时引入了全局平均池化
    def forward(self,x):
        x=self.features(x)
        x = self.avgpool(x) #TODO:这一块什么时候引入的
        x=x.flatten(x,1)
        x=x.classifier(x)
        return x

def inception_1(x,input,output):
    feature=nn.Sequential(
        nn.Conv2d(in_channels=input,out_channels=output,kernel_size=1),
        nn.ReLU(inplace=True),
    )
    return feature(x)
def inception_3(x,input,_output,output):
    feature=nn.Sequential(
        nn.Conv2d(in_channels=input,out_channels=_output,kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=_output,out_channels=output,kernel_size=3,padding=1),
        nn.ReLU(inplace=True),
    )
    return feature(x)
def inception_5(x,input,_output,output):
    feature=nn.Sequential(
        nn.Conv2d(in_channels=input,out_channels=_output,kernel_size=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=_output,out_channels=output,kernel_size=5,padding=2),
        nn.ReLU(inplace=True),
    )
    return feature(x)
def inception_MaxPool(x,input,output):
    feature=nn.Sequential(
        nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
        nn.Conv2d(in_channels=input,out_channels=output,kernel_size=1),
        nn.ReLU(inplace=True),
    )
    return feature(x)
def inception(x,input,output1,_output3,output3,_output5,output5,_outputM):
    x_1=inception_1(x,input,output1)
    x_3=inception_3(x,input,_output3,output3)
    x_5=inception_5(x,input,_output5,output5)
    x_M=inception_MaxPool(x,input,_outputM)
    outputs=[x_1,x_3,x_5,x_M]       #TODO:注意这里的聚合操作
    outputs=torch.cat(outputs,1)

    return outputs


class GoogLeNet_v1(nn.Module):
    def __init__(self,num_classes):
        super(GoogLeNet_v1,self).__init__()
        self.num_classes=num_classes
        self.features1=nn.Sequential(
            nn.Conv2d(3,64,7,stride=2,padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3,stride=2),
            nn.Conv2d(64,64,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,192,3,stride=1),
            nn.MaxPool2d(3,stride=2),
        )
        self.maxpool=nn.MaxPool2d(3,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc=nn.Linear(in_features=1024*1*1,out_features=self.num_classes)
        self.dropout=nn.Dropout()
    def forword(self,x):
        x=self.features1(x)
        x=inception(x,192,64,96,128,16,32,192,32) #3a
        x=inception(x,256,128,128,192,32,96,64) #3b
        x=self.maxpool(x)
        x=inception(x,480,192,96,208,16,48,64) #4a
        x=inception(x,512,160,112,224,24,64,64) #4b
        x=inception(x,512,128,128,256,4,64,64) #4c
        x=inception(x,512,112,144,288,32,64,64) #4d
        x=inception(x,528,256,160,320,32,128,128) #4e
        x=self.maxpool(x)
        x=inception(x,832,256,160,320,32,128,128) #5a
        x=inception(x,832,384,92,384,48,128,28) #5b
        #接入全局平均池化层
        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.dropout(x)
        x=self.fc(x)

        return x




if __name__=="__main__":
    # net=Net()
    # print(net.fc1)
    # alexnet=Alex_Net()
    # print(alexnet.features)
    # vgg16=VGG_Net()
    # print(vgg16)
    googLeNet=GoogLeNet_v1(1000)
    print(googLeNet)