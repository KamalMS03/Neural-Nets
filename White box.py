#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
class nn:
    def __init__(self,ip,op):
        self.ip=ip
        self.op=op
        print(ip.shape,op.shape)
        self.l1_size=200
        self.w1=np.random.rand(self.l1_size,self.ip.shape[1]+1).T*0.3-0.15
        self.w2=np.random.rand(self.op.shape[1],self.l1_size+1).T*0.3-0.15
        print(self.w1.shape,self.w2.shape)
        for i in range(50):
            print(i)
            self.train()
  


# In[2]:



  def swish(self,x):
      return x*(1/(1+np.exp(-x)))
#Swish function - f(x)=x*swishmoid(x)
 


# In[3]:


def train(self):
       #===front propogation===#
       ip=np.append(np.ones((len(self.ip),1)),self.ip,axis=1)
       l1=self.swish(ip@self.w1)
       l1=np.append(np.ones((len(l1),1)),l1,axis=1)
       pred=self.swish(l1@self.w2)
       #===back propogation===#
       #error at each layer
       d3 = (pred - self.op)
       d2 = (d3 @ self.w2.T * l1 * (1-l1))
       d2 = d2[:,1:]
       #altering weights
       self.w2-= (l1.T @ d3)/len(ip)
       self.w1-= (ip.T @ d2)/len(ip)
       


# In[4]:


def predict(self,ip):
     ip=np.append(np.ones((len(ip),1)),ip,axis=1)
     l1=self.swish(ip@self.w1)
     l1=np.append(np.ones((len(l1),1)),l1,axis=1)
     pred=self.swish(l1@self.w2)
     return pred


# In[5]:


def accuracy(ytst,ypred):
    print("mean error:",np.mean(np.abs((ypred-ytst).astype(bool))))
    print("rmse:",np.sqrt(((ypred-ytst).astype(bool)**2).mean()))


# In[6]:


data = pd.read_csv(r"C:\Users\kamal\Downloads\archive (1)\IRIS.csv")

x = data['sepal_length']
x = x / 255
y_lab = data['species']
y=np.zeros((len(x),10))
for i in range(len(x)):
    y[i,int(y_lab[i])]=1
xtr = x[:60000, :]
ytr = y[:60000]
xtst = x[60000:, :]
ytst = y[60000:]
ytst_lab = y_lab[60000:]
net=nn(xtr,ytr)
pred=net.predict(xtst)
pred_lab = np.argmax(pred,axis=1)
accuracy(ytst_lab,pred_lab)


# In[ ]:




