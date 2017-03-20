# Configuration of this job
parser = argparse.ArgumentParser()
# Start by creating a new config file and changing the line below
parser.add_argument('-C', '--config',default="MEDNN/ScanConfig.py")

parser.add_argument('-L', '--LoadModel',default=False)
parser.add_argument('--gpu', dest='gpuid', default="")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--NoTrain', action="store_true")
parser.add_argument('--NoAnalysis', action="store_true")
parser.add_argument('--NoData', action="store_true")
parser.add_argument('--Test', action="store_true")
parser.add_argument('-s',"--hyperparamset", default="0")
parser.add_argument('-m',"--mode", default="3")
parser.add_argument('--generator', action="store_true")

# Configure based on commandline flags... this really needs to be cleaned up
args = parser.parse_args()
Train = not args.NoTrain
Analyze = not args.NoAnalysis
ReadData = not args.NoData
TestMode = args.Test
UseGPU = not args.cpu
gpuid = args.gpuid
if args.hyperparamset:
    HyperParamSet = int(args.hyperparamset)

Mode = int(args.mode)

ConfigFile = args.config
useGenerator = args.generator

LoadModel=args.LoadModel

# Configuration from PBS:
if "PBS_ARRAYID" in os.environ:
    HyperParamSet = int(os.environ["PBS_ARRAYID"])

if "PBS_QUEUE" in os.environ:
    if "cpu" in os.environ["PBS_QUEUE"]:
        UseGPU=False
    if "gpu" in os.environ["PBS_QUEUE"]:
        UseGPU=True
        gpuid=int(os.environ["PBS_QUEUE"][3:4])

TheanoConfig=""
        
if TestMode:
    TheanoConfig+="optimizer=fast_compile,exception_verbosity=high,"
if UseGPU:
    print "Using GPU",gpuid
    TheanoConfig+="mode=FAST_RUN,device=gpu"+str(gpuid)+",floatX=float32,force_device=True"
else:
    print "Using CPU."

os.environ['THEANO_FLAGS'] = TheanoConfig
