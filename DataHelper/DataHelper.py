#!/usr/bin/env python
# coding: utf-8

# In[1]:

def load_images(path,valid_split=0):
    import numpy as np,pandas as pd
    if(valid_split>1 or valid_split<0):
        raise ValueError("Invalid split size")
    import os
    from PIL import Image as im
    if(path[-1]!="/"):
        path+="/"
    images = []
    for folder in os.listdir(path):
        subfolder_images=[]
        subpath=path+folder
        for filename in os.listdir(subpath):
            img = im.open(subpath+ '/' + filename)
            img=np.asarray(img)
            subfolder_images.append(img)
        subfolder_images = np.array(subfolder_images)
        images.append(subfolder_images)
    images=np.array(images)
    length=images.shape[0]
    size_valid=int(length*valid_split)
    return images[size_valid:],images[:size_valid]

def CreateNImages(direc,name,todir,height,width):
    import pandas as pd
    import numpy as np
    from PIL import Image as im
    import string
    import os
    ltrs=string.ascii_lowercase
    img = im.open(direc+"/"+name)
    size=img.size
    c=0
    for i in range(0,size[1]//height):
        top=i*height
        bot=top+height
        for j in range(0,size[0]//width):
            left=j*width
            right=left+width
            box=(left,top,right,bot)
            img1=img.crop(box)
            if(c<25):
                img1.save(todir+ltrs[c]+name)
            else:
                times=((c-24)//10)+1
                noz="z"*times
                totstr=todir+noz+str((c-24)%10)+name
                img1.save(totstr)
            c+=1

def loadAnImage(path,height,width):
        import os
        import numpy as np
        from PIL import Image as im
        img = im.open(path)
        size=img.size
        c=0
        arr=[]
        for i in range(0,size[1]//height):
            top=i*height
            bot=top+height
            for j in range(0,size[0]//width):
                left=j*width
                right=left+width
                box=(left,top,right,bot)
                img1=img.crop(box)
                arr.append(np.asarray(img1))
        return np.array(arr)
def showImgarr(arr,height,width):
    import numpy as np
    import matplotlib.pyplot as plt
    import numpy as np
    final_image = np.zeros((height, width, 3), dtype=np.uint8)
    h=arr[0].shape[0]
    w=arr[0].shape[1]
    hr=height//h 
    wr=width//w
    for i in range(hr):
        for j in range(we):
            idx = i * hr + j
            if idx < arr.shape[0]:
                final_image[i*h:(i+1)*h, j*w:(j+1)*w, :] = arr[idx]
    plt.imshow(final_image)
    plt.show()

def shuffleAnImage(arr,shuffles):
    import random 
    import numpy as np,pandas as pd
    from numpy import copy,append
    imgcopy=copy(arr)
    for i in range(shuffles):
        ele1=random.randint(0,35)
        ele2=random.randint(0,35)
        imgcopy[ele1],imgcopy[ele2]=imgcopy[ele2].copy(),imgcopy[ele1].copy()
    return imgcopy
def findmaxind(arr):
    maxi=-1e5
    ind=0
    ci=0
    for i in arr:
        if(i>maxi):
            ind=ci
            maxi=i
        ci+=1
    return ind
def convertBinaryMatrix(P):
    #return list
    import numpy as np
    length=P.shape[0]
    r=[0]*length
    Pn=[]
    for i in range(length):
        Pn.append(r.copy())
    for i in range(length):
        Pn[i][findmaxind(P[i])]=1
    return Pn
def findNearestZero(d,j):
    length=len(d.keys())
    l=j-1
    r=j+1
    while True:
        if(l>=0):
            if(d[l]==0):
                return l
            else:
                l-=1
        if(r<length):
            if(d[r]==0):
                return r
            else:
                r+=1
def formatPermMatrix(P):
    P=convertBinaryMatrix(P)
    #Note down number of ones in each column in a dictionary d
    #count of improperly placed ones =c
    d={}
    c=0
    for i in range(len(P)):
        d[i]=0
        for j in range(len(P)):
            if(P[j][i]==1):
                d[i]+=1
            if(d[i]>1):
                c+=1
    # a in range 0 to c
    for a in range(c):
        j=-1
        # find out first occ of more than 1 one, set as j
        for i in range(len(P)):
            if(d[i]>1):
                j=i
                break
        # Traverse column j
        f=0
        for i in range(len(P)):
            if(P[i][j]==1):
                f+=1
            #while traversing column if encounter second 1 at Pij,start looking to nearest columns with no ones in d
            if(f>1):
                k=findNearestZero(d,j)
                # go to that column in same row(i,k) and make Pij 0 ,Pik 1 and d[k]+=1, d[j]-=1
                P[i][j]=0
                P[i][k]=1
                d[k]+=1
                d[j]-=1
                f-=1
                # if d[j]==1 go to next j where d[j]>1
                # else continue in same column
    return P
def createImage(shuffled,P):
    P=formatPermMatrix(P)
    from numpy import copy
    img=copy(shuffled)
    for i in range(len(P)):
        ind=P[i].index(1)
        img[ind]=shuffled[i].copy()
    return img
def CreateDataset(direc,height,width,extension=".jpg"):
    import os
    for name in os.listdir(direc):
        st=name[:-(len(extension)+1)]
        todir=direc+"/"+st+"/"
        path = os.path.join(direc, st)
        os.mkdir(path)
        CreateNImages(direc,name,todir,height,width)   
def AugmentedData(x,y,number,augments_per_image=8):
    import random 
    import numpy as np,pandas as pd
    from numpy import copy,append
    size=x.shape[0]
    for i in range(number):
        key=random.randint(0,size-1)
        ycopy=copy(y[key])
        imgcopy=copy(x[key])
        for j in range(augments_per_image):
            innersize=imgcopy.shape[0]
            ele1=random.randint(0,innersize-1)
            ele2=random.randint(0,innersize-1)
            imgcopy[ele1],imgcopy[ele2]=imgcopy[ele2].copy(),imgcopy[ele1].copy()
            ycopy[ele1],ycopy[ele2]=ycopy[ele2].copy(),ycopy[ele1].copy()
        x=append(x,[imgcopy],axis=0)
        y=append(y,[ycopy],axis=0)
    return x,y
def PermutationMatrix(path):
    import pandas as pd
    import numpy as np
    df=pd.read_csv(path)
    lbls = df.iloc[:,-1].values
    lbl=[]
    for i in lbls:
        x=i[1:-1]
        x=x.split(",")
        l=[int(a) for a in x]
        lbl.append(np.array(l))
    i = 0
    arr=[]
    mats=[]
    for j in range(len(lbl)):
        arr=[]
        i=0
        while(i<len(lbl[j])):
            arr.append(lbl[j][i:i+36])
            i+=36
        mats.append(arr)
    y = np.array(mats)
    return y
def createSubmission(p):
    import pandas as pd
    import numpy as np
    df=pd.read_csv(p)
    names=df.iloc[:,-2:-1].values.tolist()
    pos=df.iloc[:,-1:].values.tolist()
    finall=[]
    for i in range(0,72):
        finall.append([])
    for i in pos:
        posis=i[0][1:-1].split(", ")
        for j in range(0,36):
            finall[j].append(posis[j][1])
            finall[j+36].append(posis[j][2])
    newnames=[]
    for i in names:
        newnames.append(i[0]+".jpg")
    newdf=pd.DataFrame()
    newdf["image"]=newnames
    indexes=[]
    for i in range(6):
        for j in range(6):
            indexes.append(str(i)+str(j))
    rc=[]
    for i in indexes:
        rc.append("r"+i)
    for i in indexes:
        rc.append("c"+i)
    for j in range(72):
        newdf[rc[j]]=finall[j]
    newdf.set_index("image")
    return newdf
def posFinder(pred):
    TotRows=[]
    for i in pred:
        rows=[]
        for j in i:
            y=maxind(j)
            r=y//6
            c=y%6
            rows.append(str(r)+str(c))
        TotRows.append(rows)
    return TotRows
def helpMe():
    print("List of functions availabe to be used:\n\n")
    print('''1.load_images(path,valid_split):\nTakes in 2 parameters, returns 2 numpy arrays, corresponding to training and test set. 
    
    This function is used to load the images as arrays for training and validation.
    Example usage:\ntrain,test=load_images(\"c:/images/puzzles\",0.2)
    
    ''')
    print('''2.loadAnImage(path,height,width):\nTakes in 2 parameters, returns a numpy array consisting N*M splitted image. 
    
    Splits the input image into N*M images where each piece has Height of height and Width of width ( parameters).
    Example usage:\nimg=loadAnImage(\"c:/images/puzzles/imageNo1.jpg\",50,50)
    
    ''')
    print('''3.shuffleAnImage(arr,shuffles):\nTakes in 2 parameters, returns a numpy array. 
    
    Shuffles the image pieces of input array for "shuffles" amount of times, and returns it.
    Example usage:\nimg=shuffleAnImag(arr,50)
    
    ''')
    print('''4.showImgarr(arr,height,width):\nTakes in the array and the final image's height and width as parameter, prints the image version of it. 
    
    used to display the image. pass the output from shuffleAnImage or loadAnImage or any np array resembling it, along with desired
    height and width of the final image.
    Example usage:\nshowImgarr(arr,300,300)
    
    ''')
    print('''4.showImgarr(arr,height,width):\nTakes in the array and the final image's height and width as parameter, prints the image version of it. 
    
    used to display the image. pass the output from shuffleAnImage or loadAnImage or any np array resembling it, along with desired
    height and width of the final image.
    Example usage:\nshowImgarr(arr,300,300)
    
    ''')



    
    
