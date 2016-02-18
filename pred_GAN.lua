require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
torch.setnumthreads(1)

digit = torch.load('digit.t7')
in_dim = digit[1]:size(2)

num_state = 10
act_dim = 4
T = torch.ones(num_state,act_dim)
correct = torch.ones(num_state)
for i = 1,num_state-1 do
    action = torch.random(act_dim)
    T[i][action] = i+1
    correct[i] = action
end


hid_dim = 100
out_dim = 1
--Discrim
local input = nn.Identity()()
local hid_lin = nn.Linear(in_dim,hid_dim)
local hid = nn.ReLU()(hid_lin(input))
local out_lin = nn.Linear(hid_dim,out_dim)
local output = nn.Sigmoid()(out_lin(hid))
network = nn.gModule({input},{output})
--Gen
gen_hid_dim = 100
fact_dim  = 10
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(in_dim,gen_hid_dim)(input))
local action = nn.Identity()()
local factor = nn.CMulTable(){nn.Linear(gen_hid_dim,fact_dim)(hid),nn.Linear(act_dim,fact_dim)(action)}
local last_hid = nn.ReLU()(nn.Linear(fact_dim,gen_hid_dim)(factor))
local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(last_hid))
gen_network = nn.gModule({input,action},{output})

--full
local input = nn.Identity()()
local action = nn.Identity()()
local connect= nn.GradientReversal()(gen_network{input,action})
local full_out = network(connect)
full_network = nn.gModule({input,action},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = 320
local data = torch.zeros(mb_dim,in_dim)
local target = torch.zeros(mb_dim,1)
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)

local get_data = function()
    local state = torch.Tensor(mb_dim,in_dim)
    local action = torch.zeros(mb_dim,act_dim)
    local state_prime = torch.Tensor(mb_dim,in_dim)
    local shuffle = {}
    for i=1,num_state do
        shuffle[i] = torch.randperm(digit[i]:size(1))
    end

    for i=1,mb_dim do
        local s = torch.random(num_state)
        state[i] = digit[s][shuffle[s][i] ]
        local a = torch.random(act_dim)
        action[i][a] = 1
        sPrime = T[s][a]
        state_prime[i] = digit[sPrime][shuffle[sPrime][i] ]
    end
    return state,action,state_prime
end
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:training()
    
    state,action,state_prime = get_data()
    for i=1,mb_dim/2 do
        data[i] = state_prime[i]
    end
    target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2)
    target[{{mb_dim/2+1,-1}}] = torch.zeros(mb_dim/2)

    data[{{mb_dim/2+1,-1}}]  = gen_network:forward{state[{{1,mb_dim/2}}],action[{{1,mb_dim/2}}]}

    output = network:forward(data)
    loss = bce_crit:forward(output,target)
    grad = bce_crit:backward(output,target)
    network:backward(data,grad)
    r = (output[{{1,mb_dim/2}}]:sum() + output[{{mb_dim/2+1,-1}}]:mul(-1):add(1):sum() )/mb_dim
    net_reward = net_reward + r
    return loss,dw
end
    
local train_gen = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:evaluate()

    state,action,state_prime = get_data()
    target:zero()
    local output = full_network:forward{state,action}
    local disc_loss = bce_crit:forward(output,target)
    local disc_grad = bce_crit:backward(output,target):clone()
    full_network:backward(state,disc_grad)
    local recon_loss = bce_crit:forward(gen_network.output,state_prime)
    local recon_grad = bce_crit:backward(gen_network.output,state_prime):clone()
    gen_network:backward({state,action},recon_grad)
    return disc_loss+recon_loss,dw
end
config = {
    learningRate  = 1e-3
    }
local num_steps = 1e6
local refresh = 1e3
local cumloss =0 
local plot1 = gnuplot.figure()
local plot2 = gnuplot.figure()
for i=1,num_steps do
    for k=1,1 do
        x,batchloss = optim.rmsprop(train_dis,w,config)
    end
    x,batchloss = optim.rmsprop(train_gen,w,config)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        gnuplot.figure(plot2)
        gnuplot.imagesc(data[{{mb_dim/2+1}}]:reshape(28,28))
        gnuplot.figure(plot1)
        samples1 = torch.randperm(not8:size(1))
        samples2 = torch.randperm(notnot8:size(1))
        for i=1,mb_dim do
            if i<=mb_dim/2 then
                data[i] = not8[samples1[i]]
            else
                data[i] = notnot8[samples2[i]]
            end 
        end
        output = network:forward(data)
        print(output[{{1,mb_dim/2}}]:mean(),output[{{mb_dim/2+1,-1}}]:mean())
        net_reward = 0
        cumloss = 0
    end
end

