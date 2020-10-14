#encoding=utf-8
import torch
from send_mail import sentemail
import argparse
from train import Process


if __name__=='__main__':
    #Function(sys.argv)
    parse=argparse.ArgumentParser(description="train or test")
    parse.add_argument('--net', type=str,default='ResNet50',help='select the model to train and test')
    parse.add_argument('--pretrained', type=bool, default=True, help='if model pretrained')
    parse.add_argument('--train', type=str,default='train', help='train the model')
    parse.add_argument('--epoch', type=int, default=1, help='the epoch')
    parse.add_argument('--batch_size', type=int, default=4, help='the epoch')
    args=parse.parse_args()
    print(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    pro = Process(device,model=args.net,batch_size=args.batch_size)
    pro.train(epoch=args.epoch)
    sentemail()