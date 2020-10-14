#用来划分测试集，验证集，测试集
# 要求：
# 1.数据集是均衡的，正例：负例=1：1
# 2.数据集划分要求不重合，也就是三个集合彼此没有交集
# 3.三种集合的比例是可以调整的
# 4.生成train.txt、test.txt、val.txt三个文本文件，包含image_name 和 target

import os
import  random
import math
#输入数据名列表，数据分配比例 ，输出训练集等三个列表
def slpit_dataset(dataset,train,val,test):
    s=train+val+test
    train=train/s
    val = val / s
    test = test / s
    random.shuffle(dataset)
    O_train=dataset[0:math.ceil(len(dataset)*train)]
    O_val=dataset[math.ceil(len(dataset)*train):math.ceil(len(dataset)*(train+val))]
    O_test=dataset[math.ceil(len(dataset)*(train+val)):]

    return O_train,O_val,O_test, len(dataset),len(O_val)+len(O_test)+len(O_train)

#遍历文件夹，生成包含图像名称的列表
# 要求：
# 1.只索引相应的文件夹，并生成列表返回
# 2.返回列表和长度
def generate_name(path):
    return os.listdir(path)

#写入txt文件中
# 要求：
# 1.分别生成train.txt等三个文件
# 2.txt文件包含文件名和类别，image_name:target
def generate_txt(filename,filepath,image_name,target):
    with open(filename,mode='a+') as f:
        for i in image_name:
            f.write(filepath+'/'+i+' '+str(target)+'\n')
if __name__=='__main__':
    path='./data'
    for name in os.listdir(path):
        if name.endswith(".txt"):
            os.remove(os.path.join(path,name))
            print("Delete file:"+os.path.join(path,name))
    pos_path='./data/Positive'
    neg_path = './data/Negative'
    pos=generate_name(pos_path)
    neg=generate_name(neg_path)
    p_train_name,p_val_name,p_test_name,_,_=slpit_dataset(pos, 8, 1, 1)
    n_train_name, n_val_name, n_test_name, _, _ = slpit_dataset(neg, 8, 1, 1)
    generate_txt("./data/train.txt",'Positive', p_train_name,1)
    generate_txt("./data/train.txt",'Negative', n_train_name, 0)
    generate_txt("./data/val.txt", 'Positive',p_val_name, 1)
    generate_txt("./data/val.txt", 'Negative',n_val_name, 0)
    generate_txt("./data/test.txt",'Positive', p_test_name, 1)
    generate_txt("./data/val.txt", 'Negative',n_test_name, 0)

    print("create the dateset finished! train:%d,val:%d,test:%d"\
          %(len(p_train_name+n_train_name),len(p_val_name+n_val_name),len(p_test_name+n_test_name)))



