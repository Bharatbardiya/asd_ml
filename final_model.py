import numpy as np
from collections import defaultdict
import pandas as pd
import sys
sys.setrecursionlimit(10**6)
import random


Trees = []

class Node:
    def __init__(self , f=0):
        self.feature = f
        self.left = None
        self.right = None

class ASDModel:
        
    def __init__(self):
        self.Trees = []
            
    def entropy(self, t,f):
        if t==0 or f==0:
            return 0
        r  = t+f
        return -((t/r)*np.log2(t/r) + (f/r)*np.log2(f/r) )

    def train(self, features , rows, root,data): 
        
        pos = 0
        neg = 0
        n = len(rows)
        for r in rows:
            if data[r][-1] == 1:
                pos += 1
            else:
                neg += 1

        if not pos:
            root.feature = 0
            return 
        if not neg:
            root.feature = 1
            return 
    

        pos_p = pos/n
        neg_p = neg/n

        H = -(pos_p*np.log2(pos_p) + neg_p*np.log2(neg_p) )     
        
        FF = defaultdict(int)
        FT = defaultdict(int)
        TF = defaultdict(int)
        TT = defaultdict(int)
        

        for r in rows:
            for f in features:
                if data[r][f]==0:
                    if data[r][-1]==0:
                        FF[f] += 1
                    else:
                        FT[f] += 1
                else:
                    if data[r][-1]==0:
                        TF[f] += 1
                    else:
                        TT[f] += 1
        
        max_gain = -float('inf')
        attri = 0

        for f in features:
            true = TT[f] + TF[f]
            false = FT[f] + FF[f]
            d = true + false
            

            rem = (true/d)*self.entropy(TT[f] , TF[f]) + (false/d)*self.entropy(FT[f] , FF[f])
            
            gain  = rem - H
            if gain>max_gain:
                max_gain = gain
                attri = f

        root.feature = attri

        left , right = set() , set()
        for r in rows:
            if data[r][attri]==1:
                right.add(r)
            else:
                left.add(r)
        
        l = Node()
        r = Node()
        root.left = l
        root.right = r
        features.discard(attri)
        if not features:
            return 

        if not left or not right:
            pos = 0
            neg = 0
            for r in rows:
                if data[r][-1]==0:
                    neg+=1
                else:
                    pos += 1
            if not left:
                root.left.feature = (1 if pos>neg else 0)
                
                self.train(features , right , root.right,data)
                
            if not right:
                root.right.feature = (0 if neg>pos else 1)
                self.train(features , left , root.left,data)
            features.add(attri)
            return 
        
        self.train(features , left , root.left,data)
        self.train(features , right , root.right,data)
        features.add(attri)


    def predict(self, root,data):
        if not root.left and root.right:
            return root.feature
        x = root.feature
        if data[x]==0:
            if not root.left:
                return 1
            return self.predict(root.left , data)
        else:
            if not root.right:
                return 0
            return self.predict(root.right , data)


def main(): 
    
    arr = [i for i in range(1,33)]

    data=pd.read_csv('data.csv')

    data = data.to_numpy()
    n = len(data)
    p = int(0.8*n)
    test = data[p:] # testing data
    data = data[:p] # training data
    
    
    asd_model = ASDModel()
    
    for i in range(20): # number of trees
        random.shuffle(arr)
        features = arr[:15]
        d = features[:]
        features = set(features)
        Tree =  Node()  
        s = set([i for i in range(len(data))])
        asd_model.train(features,s,Tree,data)
        Trees.append([Tree,d])
    
    correct = helper(test, asd_model)
    print(f"\naccuracy : {round(correct/len(test)*100, 4)}% ")



      
def helper(data, asd_model):
    correct = 0
    for row in data:
        out = row[-1]
        row = row[:-1]
        #voting 
        pos = 0
        neg = 0
        for tree,arr in Trees:
            r = asd_model.predict(tree,row)
            
            if r==0:
                neg+=1
            else:
                pos+=1
        r = 1 if pos>=neg else 0
        
        if row[-1]==r:
            correct += 1
    return correct


if __name__=="__main__": 
    main()