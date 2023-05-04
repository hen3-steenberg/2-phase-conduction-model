import numpy as np
import matplotlib.pyplot as plt



def CreateFuzzy(Values, Locations, SteppingSpeeds):
    Locs = np.array(Locations)
    #print(Locations)
    Aggro = np.array(SteppingSpeeds)
    #print(Aggro)
    Vals = np.array(Values)
    def Inner(func):
        def fuzzy(x):
            params = np.array(x)
            X = np.broadcast_to(params, (Locs.size, params.size)).T
            Normalized = (X - Locs)
            Normalized = Aggro * Normalized
            Inner = func(Normalized)
            Weights = np.append(np.ones((params.size,1)),Inner,axis=1) - np.append(Inner,np.zeros((params.size,1)),axis=1)
            #Size = np.sum(Weights,axis=1,keepdims=True)
            #print(Size)
            #Weights = Weights / Size
            Out = Vals * Weights
            return np.sum(Out,axis=1)
        return fuzzy
    return Inner
    
@CreateFuzzy([0,1,0.5],[-5, 5],[5,5])    
def sigmoid(x):
    return 1/(1 + np.exp(-x))

@CreateFuzzy([0,0.02,0],[-5, 5],[5,5])
def area(x):
    return np.log(1 + np.exp(x))

        
X = np.arange(-10,20,0.1)
Y = sigmoid(X)
A = area(X)
plt.plot(X,Y)
plt.plot(X,A)
plt.show()