require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
--torch.setnumthreads(1)
T = torch.range(2,9):cat(torch.ones(1))

num_state = 10
in_dim = num_state 

hid_dim = 100
out_dim = 1
dropout_p = .1
fake_p = .5
--rev_grad = true
gen_hid_dim = 100
--Discrim
local input = nn.Identity()()
hid = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(in_dim,hid_dim)(input)))
hid2 = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid)))
--one more layer here?
local out_lin = nn.Linear(hid_dim,out_dim)
local output = nn.Sigmoid()(out_lin(hid2))
network = nn.gModule({input},{output})
--Gen
noise_dim = 10
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
local last_hid = nn.ReLU()(nn.Linear(gen_hid_dim,gen_hid_dim)(hid))
local output =nn.Sigmoid()(nn.Linear(gen_hid_dim,in_dim)(last_hid))
gen_network = nn.gModule({input},{output})
--full
local input = nn.Identity()()
if rev_grad then
    connect= nn.GradientReversal()(gen_network(input))
else
    connect= gen_network(input)
end
local full_out = network(connect)
full_network = nn.gModule({input},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = num_state*1
local target = torch.zeros(mb_dim,1)

local s
local noise_level = .25
local get_data = function(dim)
    local state = torch.rand(dim,in_dim):mul(noise_level)

    for i=1,dim do
        s = torch.random(num_state)
        state[i][s] = 1-torch.rand(1):mul(noise_level)[1]
    end
    return state
end
local get_noise = function(dim)
    local noise = torch.zeros(dim,noise_dim)
    for i=1,dim do
        noise[i][torch.random(noise_dim)] = 1
    end
    return torch.randn(dim,noise_dim)
    --return noise
end
local get_complete_data = function()
    return torch.eye(num_state)
end
local data = torch.zeros(mb_dim,in_dim)
num_real = math.floor(mb_dim*(1-fake_p))
num_fake = mb_dim - num_real
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:training()
    
    state = get_data(num_real)
    for i=1,num_real do
        data[i] = state[i]
    end
    target[{{1,num_real}}] = torch.ones(num_real)
    target[{{num_real+1,-1}}] = torch.zeros(num_fake)

    data[{{num_real+1,-1}}]  = gen_network:forward(get_noise(num_fake))
    

    output = network:forward(data)
    loss = bce_crit:forward(output,target)
    grad = bce_crit:backward(output,target)
    network:backward(data,grad)
    r = (output[{{1,num_real}}]:sum() + output[{{num_real+1,-1}}]:mul(-1):add(1):sum() )/mb_dim
    net_reward = net_reward + r
    return loss,dw
end
    
local train_gen = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:evaluate()

    if rev_grad then
        target:zero()
    else
        target:zero():add(1)
    end
    local data = get_noise(mb_dim)
    local output = full_network:forward(data)
    local disc_loss = bce_crit:forward(output,target)
    local disc_grad = bce_crit:backward(output,target):clone()
    full_network:backward(data,disc_grad)
    return disc_loss,dw
end
config_dis = {
    learningRate  = 1e-3
    }
config_gen = {
    learningRate  = 1e-3
    }
local num_steps = 1e6
local refresh = 1e3
local cumloss =0 
local plot1 = gnuplot.figure()
local plot2 = gnuplot.figure()
for i=1,num_steps do
    for k=1,5 do
        x,batchloss = optim.adam(train_dis,w,config_dis)
    end
    x,batchloss = optim.adam(train_gen,w,config_gen)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        gnuplot.figure(plot1)
        gnuplot.hist(full_network:forward(get_noise(mb_dim)))
        print(gen_network.output[1])
        gnuplot.axis{0,1,0,20}

        gnuplot.figure(plot2)
        --gnuplot.bar(gen_network.output[1])
        gnuplot.imagesc(gen_network.output)
        gnuplot.plotflush()
        net_reward = 0
        cumloss = 0
    end
end

