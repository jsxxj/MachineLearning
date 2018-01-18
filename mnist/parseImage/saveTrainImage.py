#encoding=utf-8
import struct
import numpy as np
import  matplotlib.pyplot as plt
import Image
#二进制的形式读入
filename='train-images.idx3-ubyte'
binfile=open(filename,'rb')
buf=binfile.read()
#大端法读入4个unsigned int32
#struct用法参见网站 http://www.cnblogs.com/gala/archive/2011/09/22/2184801.html

index=0
magic,numImages,numRows,numColumns=struct.unpack_from('>IIII',buf,index)
index+=struct.calcsize('>IIII')
#将每张图片按照格式存储到对应位置
for image in range(0,numImages):
    im=struct.unpack_from('>784B',buf,index)
    index+=struct.calcsize('>784B')
   #这里注意 Image对象的dtype是uint8，需要转换
    #im=np.array(im,dtype='uint8')
    #im=im.reshape(28,28)
    #fig=plt.figure()
    #plotwindow=fig.add_subplot(111)
    #plt.imshow(im,cmap='gray')
    #plt.show()
    im=Image.fromarray(im)
    im.save('train/train_%s.bmp'%image,'bmp')
    
    
 
方法2：
#encdoing=UTF-8
from PIL import Image
import struct

def read_image(filename):
  filename='train-images-idx3-ubyte'
  f=open(filename,'rb')
  index=0
  buf=f,read()
  f.close()
  
  magic,images,rows,columns=struct.unpack_from('>IIII',buf,index)
  index+=struct.calcsize('>IIII')
  for i in xrange(images):
    image=Image.new('L',(columns,rows))
    for x in xrange(rows):
      for y in xrange(columns):
         image.putpixel((y,x),int(struct.unpack_from('>B',buf,index)[0]))
         index+=struct.calcsize('>B')
  print 'save'+str(i)+'image'
  image.save('home/save/'+str(i)+'.png')
  print ok

