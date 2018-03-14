#encoding:utf-8
#adapting from https://github.com/guohuifengby/pythonConvertCifar-10/blob/master/convert.py

#conver cifar10 into schaema sutiable for 
#https://github.com/tensorflow/models/blob/master/research/inception/inception/data/build_image_data.py
"""
The image data set is expected to reside in JPEG files located in the
following directory structure.
  data_dir/label_0/image0.jpeg
  data_dir/label_0/image1.jpg
  ...
  data_dir/label_1/weird-image.jpeg
  data_dir/label_1/my-image.jpeg
  ...
"""

from scipy.misc import imsave
import numpy as np
import os


# 解压缩，返回解压后的字典
def unpickle(file):
    import _pickle as cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo,encoding='iso-8859-1')
    fo.close()
    return dict

if __name__ == '__main__':

       
    base_path = 'cifar-10-batches-py' #base path where to store data_batch* and test_batch
    train_dir = os.path.join(base_path,"train") #path to store recover taringning data
    validate_dir = os.path.join(base_path,"validate")#path to store recover testing data
    classes = 10 
    
    # create dictionary to store data
    for i in range(0,classes):
        train_lable_dir = os.path.join(train_dir,'label_'+str(i))
        validate_lable_dir = os.path.join(validate_dir,'label_'+str(i))
        if not os.path.exists(train_lable_dir):
            os.makedirs(train_lable_dir)
        if not os.path.exists(validate_lable_dir):
            os.makedirs(validate_lable_dir)    
        
    # 生成训练集图片，如果需要png格式，只需要改图片后缀名即可。
    for j in range(1, 6):
        train_data_name = os.path.join(base_path,"data_batch_" + str(j)) # read data_batch* file
        print(train_data_name)
        Xtr = unpickle(train_data_name)
        print(train_data_name + " is loading...")
        for i in range(0, 10000):
            img = np.reshape(Xtr['data'][i], (3, 32, 32))  # Xtr['data']为图片二进制数据
            img = img.transpose(1, 2, 0)  # 读取image
            lable_dir = os.path.join(train_dir,'label_'+str(Xtr['labels'][i]))
            pic_name = os.path.join(lable_dir,str(i + (j - 1)*10000) + '.jpg')  # Xtr['labels']为图片的标签，值范围0-9
            imsave(pic_name, img)
        print(train_data_name + " loaded.")
    
    print("test_batch is loading...")
    
    # 生成测试集图片
    validate_data_name = os.path.join(base_path,'test_batch')
    testXtr = unpickle(validate_data_name)
    for i in range(0, 10000):
        img = np.reshape(testXtr['data'][i], (3, 32, 32))
        img = img.transpose(1, 2, 0)
        lable_dir = os.path.join(validate_dir,'label_'+str(testXtr['labels'][i]))
        pic_name = os.path.join(lable_dir,str(i + (j - 1)*10000) + '.jpg')
        imsave(pic_name, img)
    print(validate_data_name + " loaded.")
    