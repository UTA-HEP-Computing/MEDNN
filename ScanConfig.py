import random
import getopt
from DLTools.Permutator import *
import sys,argparse

# Generation Model
Config={
    "GenerationModel":"'Load'",
    "MaxEvents":1e6,
    "FractionTest":0.1,

    "Y_min":-150,
    "Y_max":200,

    "Sigma":0.,

    "Epochs":1000,
    "BatchSize": 2048*8,
    
    "LearningRate":0.005,
    
    "WeightInitialization":"'normal'",

    "Width":32,
    "Depth":1
    }


# The Modes correspond to different models
if Mode==1:
    print "Classification Mode."
    Config.update({
        "Mode":"'Classification'",
        "NBins":1000,
        "loss":"'categorical_crossentropy'",
        "optimizer":"'rmsprop'"})
    Params={ "Width":[32,64,128,256,512],
         "Depth":range(1,10),
          }

if Mode==2:
    print "Highway Mode."
    Config.update({
        "Mode":"'Highway'",
        "loss":"'mse'",
        "optimizer":"'adam'",
        "N_Highways":10,
        "Epochs":1000
        })
    
if Mode==3:
    print "Regression Mode."
    Config.update({
        "Mode":"'Regression'",
        "loss":"'GaussianNLL'",
        "optimizer": "'Adam'" })

# Define the Name
Name="MEDNN_"+eval(Config["Mode"])

# If we are doing a scan...
if "Params" in dir():
    PS=Permutator(Params)
    Combos=PS.Permutations()

    print "HyperParameter Scan: ", len(Combos), "possible combiniations."
    
    if "HyperParamSet" in dir():
        i=int(HyperParamSet)
    else:
        # Set Seed based on time
        random.seed()
        i=int(round(len(Combos)*random.random()))
        print "Randomly picking HyperParameter Set"

    print "Picked combination: ",i

    for k in Combos[i]:
        Config[k]=Combos[i][k]

    for MetaData in Params.keys():
        val=str(Config[MetaData]).replace('"',"")
        Name+="_"+val.replace("'","")

    if "TestMode" in dir() and TestMode:
        print "Test mode... running on reduced dataset."
        Config["MaxEvents"]=1e4
        Config["Epochs"]=3

# Done... 
print "Model Filename: ",Name

