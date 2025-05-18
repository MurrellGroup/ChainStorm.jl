#If you need the animations via GLMakie to run headless, in linux you can install xvfb, then run these in the terminal:
#Xvfb :99 -screen 0 1024x768x24 &
#export DISPLAY=:99
#Then launch Julia and run this at the top:
#get!(ENV, "DISPLAY", ":99")

using Pkg
pkg"registry add https://github.com/MurrellGroup/MurrellGroupRegistry"
Pkg.activate(".")
Pkg.add(["JLD2", "Flux", "GLMakie", "ProtPlot"])
Pkg.add(url = "https://github.com/MurrellGroup/ChainStorm.jl")
Pkg.add(["CUDA", "cuDNN"]) #<- If GPU

using ChainStorm, JLD2, Flux
using GLMakie, ProtPlot

#GPUnum = 0                              #<-To limit the run to one particular GPU
#ENV["CUDA_VISIBLE_DEVICES"] = GPUnum    #<-To limit the run to one particular GPU
using CUDA
#device!(0)                              #<-To limit the run to one particular GPU
device = gpu 
#device = identity                       #<- If no GPU

!("ChainStorm_chkpt_3.jld2" in readdir()) && run(`wget https://huggingface.co/MurrellLab/ChainStorm/resolve/main/ChainStorm_chkpt_3.jld2`)
model_state = JLD2.load("ChainStorm_chkpt_3.jld2", "model_state");
loadedmodel = FlowcoderSC(384, 6, 6);
Flux.loadmodel!(loadedmodel, model_state);
testmode!(loadedmodel);
model = loadedmodel |> device;

chainlengths = [124,124] #<- The model's only input
b = dummy_batch(chainlengths)
paths = ChainStorm.Tracker()
g = flow_quickgen(b, model, d = device, tracker = paths) #<- Model inference call
id = join(string.(chainlengths),"_")*"-"*join(rand("0123456789ABCDEFG", 4))
export_pdb("$(id).pdb", g, b.chainids, b.resinds) #<- Save PDB
samp = gen2prot(g, b.chainids, b.resinds)
animate_trajectory("$(id).mp4", samp, first_trajectory(paths), viewmode = :fit, size = (1280, 720), framerate = 25) #<- Animate design process
