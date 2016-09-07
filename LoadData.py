import h5py
import numpy as np
from keras.utils import np_utils

def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def GaussianDiagonalMatrix(min,max,bins,sigma):
    binwidth=(max-min)/float(bins)
    x=np.arange(min,max,binwidth)
    Matrix=[]

    Middle=gaussian(x,binwidth*bins/2,sigma)
    Norm=sum(Middle)

    for i in x:
        Matrix+=[gaussian(x,i,sigma)/Norm]

    return np.matrix(Matrix)


def BinVariable(y,bins,min,max,Sigma=0.):
    binwidth=(max-min)/float(bins)

    Y= np_utils.to_categorical(np.digitize(y,bins=np.arange(min,max,binwidth)),bins)
    
    print Y.shape

    if Sigma==0.:
        return Y

    M=GaussianDiagonalMatrix(min,max,bins,Sigma)

    return np.transpose(np.dot(M,np.transpose(Y)))



def LoadData(filenames,FractionTest=.1,MaxEvents=-1, MinEvents=-1, Shuffle=False, Bin=False, N_Inputs=15):
    FHandles=[]

    First=True

    Train_X=None
    Train_Y=None

    Test_X=None
    Test_Y=None

    for filename in filenames:
        print "Reading:",filename
        f=h5py.File(filename,"r")
        FHandles+=[f]

        N_Incoming_Inputs=18

        if MaxEvents!=-1:
            try:
                print "Trying To Read Exactly", MaxEvents,"Events..."
                X_In=f[u'SherpaEvents']["Event"]["Particles"][:MaxEvents]
                print "Success."
            except:
                print "Failed... Trying to read all."
                X_In=f[u'SherpaEvents']["Event"]["Particles"]
        else:
            X_In=f[u'SherpaEvents']["Event"]["Particles"]

        X_In_Shape=X_In.shape

        N_Incoming_Inputs=X_In_Shape[1]*X_In_Shape[2]

        N=X_In_Shape[0]
        N_Test=int(round(FractionTest*N))
        N_Train=N-N_Test

        if MaxEvents!=-1:
            if MaxEvents>N:
                print "Warning: Sample",S," has",N," events which is less that ",MaxEvents,"."
                print "Using ",N_Train,"Events for training."
                print "Using ",N_Test,"Events for training."
            else:
                N_Test=int(round(FractionTest*MaxEvents))
                N_Train=MaxEvents-N_Test

        if MinEvents!=-1:
            if N_Train<MinEvents:
                print "Warning: Sample",S," has",N_Train," training events which is less that ",MaxEvents,"."

        X_train=np.array(X_In[0:N_Train]).reshape(N_Train,N_Incoming_Inputs)[:,0:N_Inputs]
        X_test=np.array(X_In[N_Train:N_Train+N_Test]).reshape(N_Test,N_Incoming_Inputs)[:,0:N_Inputs]
        
        y_train=np.array(f[u'SherpaEvents']["Event"]["weight"][0:N_Train])
        y_test=np.array(f[u'SherpaEvents']["Event"]["weight"][N_Train:N_Train+N_Test])

        if not First:
            Train_X=np.concatenate(X_train)
            Train_Y=np.concatenate(y_train)

            Test_X=np.concatenate(X_test)
            Test_Y=np.concatenate(y_test)
        else:
            Train_X=X_train
            Train_Y=y_train

            Test_X=X_test
            Test_Y=y_test
            First=False        



    if Shuffle:
        Train_X,Train_Y=shuffle_in_unison_inplace(Train_X,Train_Y)
        Test_X,Test_Y=shuffle_in_unison_inplace(Test_X,Test_Y)

    if Bin:
        return (Train_X, BinVariable(Train_Y,Bin[0],Bin[1],Bin[2],Bin[3])), (Test_X, BinVariable(Test_Y,Bin[0],Bin[1],Bin[2],Bin[3])), (Train_Y, Test_Y)

        
    return (Train_X, Train_Y), (Test_X, Test_Y)



