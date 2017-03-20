import sys,os,argparse

execfile("MEDNN/Arguments.py")
from keras.callbacks import EarlyStopping

# Process the ConfigFile
execfile(ConfigFile)

# Now put config in the current scope. Must find a prettier way.
if "Config" in dir():
    for a in Config:
        exec(a+"="+str(Config[a]))

# Load the Data
from MEDNN.LoadData import *

if Mode=="Regression" or Mode=="Highway":
    Binning=False
if Mode=="Classification":
    Binning=[NBins,Y_min,Y_max,Sigma]

# Used for Debugging... Feed fake data... all zeros.
if not ReadData:  
    N_Train=int((1.-FractionTest)*float(MaxEvents))
    N_Test=int(FractionTest*float(MaxEvents))

    Train_X=np.zeros((N_Train,15))
    Train_Y=np.zeros((N_Train,))

    Test_X=np.zeros((N_Test,15))
    Test_Y=np.zeros((N_Test,))

if ReadData:
    from MEDNN.InputFiles import InputFiles

    if Binning:
        (Train_X, Train_Y),(Test_X, Test_Y), (Train_YT, Test_YT) = LoadData(InputFiles,
                                                                            FractionTest=FractionTest,
                                                                            MaxEvents=MaxEvents,
                                                                            MinEvents=-1,
                                                                            Shuffle=False,
                                                                            Bin=Binning)
    else:
        (Train_X, Train_Y),(Test_X, Test_Y)= LoadData(InputFiles,
                                                      FractionTest=FractionTest,
                                                      MaxEvents=MaxEvents,
                                                      MinEvents=-1,
                                                      Shuffle=False,
                                                      Bin=Binning)

        M=np.mean(Train_Y)
        V=np.var(Train_Y)
        
        # Normalize Y to be between [0,1]
        #Y_max=np.max(Train_Y)
        #Y_min=np.min(Train_Y)
        #MassNorm=(Y_max-Y_min)
        #Train_YN=(Train_Y-Y_min)/MassNorm
        #Test_YN=(Test_Y-Y_min)/MassNorm
        
        # Normalize Y by shifting by mean and dividing by variance 
        #if V != 0.:
        #    Train_YN=(Train_Y-M)/V+.5
        #    Test_YN=(Test_Y-M)/V+.5

        # Don't Normalize
        Train_YN=Train_Y
        Test_YN=Test_Y
        
        # Custom Loss Function
        # y_true and y_pred must be same shape. Pad y_true with dummy values so that y_pred can include the error (I think)
        if loss=="GaussianNLL":
            Train_YN = np.hstack((Train_YN, 2.*np.ones_like(Train_YN))).reshape((-1,2)) 
            Outputsize=2
        else:
            Outputsize=1


# Normalize the Data... seems to be critical!
Norm=np.max(Train_X)
if Norm != 0.:
    Train_X=Train_X/Norm
    Test_X=Test_X/Norm



# Build/Load the Model
from DLModels.Regression import *
from DLTools.ModelWrapper import ModelWrapper

NInputs=15

# Instantiate the Model

if Mode=="Regression":
    MyModel=FullyConnectedRegression(Name,NInputs,Width,Depth,WeightInitialization,Outputsize)
if Mode=="Classification":
    MyModel=FullyConnectedClassification(Name,NInputs,Width,Depth,Binning[0],WeightInitialization)
if Mode=="Highway":
    MyModel=HighwayRegression(Name,NInputs,Width,Depth,WeightInitialization,N_Highways)
    Mode="Regression"

if LoadModel:
    print "Loading Model From:",LoadModel
    if LoadModel[-1]=="/":
        LoadModel=LoadModel[:-1]
    Name=os.path.basename(LoadModel)
#    MyModel=ModelWrapper(Name)
#    MyModel.InDir=os.path.dirname(LoadModel)
    MyModel.InDir=LoadModel
    MyModel.Load()
    MyModel.Initialize()
    MyModel.MakeOutputDir()
else:
    # Build it
    MyModel.Build()

# Print out the Model Summary
MyModel.Model.summary()

# Compile The Model
print "Compiling Model."
MyModel.Compile(Loss=loss,Optimizer=optimizer) 

# Train
if Train:
    print "Training."

    #callbacks=[EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='min') ]
    callbacks=[]

    if Mode=="Regression":
        MyModel.Train(Train_X, Train_YN, Epochs, BatchSize,Callbacks=callbacks)
        score = MyModel.Model.evaluate(Test_X, Test_YN, batch_size=BatchSize)

    if Mode=="Classification":
        MyModel.Train(Train_X, Train_Y, Epochs, BatchSize, Callbacks=callbacks)
        score = MyModel.Model.evaluate(Test_X, Test_Y, batch_size=BatchSize)

    print
    print "Final Score:", score

# Save Model
if Train:
    MyModel.Save()

# Analysis
if Analyze:
    from DLAnalysis.Regression import *
    if Mode=="Regression":
        [resultHist, targetHist, residualHist], result=RegressionAnalysis(MyModel,Test_X,Test_Y,Y_min,Y_max,M,V,BatchSize,CorrectY=False)

    if Mode=="Classification":
        ClassificationAnalysis(MyModel,Test_X,Test_Y,Test_YT,Y_min,Y_max,NBins,BatchSize)

# python -im MEDNN.Experiment -m 3 --NoTrain -L TrainedModels/MEDNN_\'Regression\'_32_1
