require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
torch.setnumthreads(1)

num_state = 10
in_dim = num_state 

act_dim = 4
T = torch.ones(num_state,act_dim)
correct = torch.ones(num_state)
for i = 1,num_state-1 do
    action = torch.random(act_dim)
    T[i][action] = i+1
    correct[i] = action
end


hid_dim = 100
fact_dim  = 10
out_dim = 1
--Err Pred Net
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
local action = nn.Identity()()
local factor = nn.CMulTable(){nn.Linear(hid_dim,fact_dim)(hid),nn.Linear(act_dim,fact_dim)(action)}
local hid2 = nn.ReLU()(nn.Linear(fact_dim,hid_dim)(factor))
local sPrime = nn.Identity()()
local factor2 = nn.CMulTable(){nn.Linear(hid_dim,fact_dim)(hid2),nn.Linear(in_dim,fact_dim)(sPrime)}
--one more layer here?
local out_lin = nn.Linear(fact_dim,out_dim)
local output = nn.Sigmoid()(out_lin(factor2))
network = nn.gModule({input,action,sPrime},{output})
--Gen
gen_hid_dim = 100
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
--local connect= gen_network{input,action}
local full_out = network{input,action,connect}
full_network = nn.gModule({input,action},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = 320
local target = torch.zeros(mb_dim,1)
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)

local s,a,sPrime
local get_data = function(dim)
    local state = torch.zeros(dim,in_dim)
    local action = torch.zeros(dim,act_dim)
    local state_prime = torch.zeros(dim,in_dim)

    for i=1,dim do
        s = torch.random(num_state-2)
        state[i][s] = 1
        a = torch.random(act_dim)
        action[i][a] = 1
        sPrime = T[s][a]
        state_prime[i][sPrime] = 1
    end
    return state,action,state_prime
end
local get_complete_data = function()
    local dim = num_state*act_dim
    local state = torch.zeros(dim,in_dim)
    local action = torch.zeros(dim,act_dim)
    local state_prime = torch.zeros(dim,in_dim)

    for a=1,act_dim do
        for s=1,num_state do
            local i = num_state*(a-1)+s
            state[i][s] = 1
            action[i][a] = 1
            sPrime = T[s][a]
            state_prime[i][sPrime] = 1
        end
    end
    return state,action,state_prime
end
local data = torch.zeros(mb_dim,in_dim)
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:training()
    
    state,action,state_prime = get_data(mb_dim/2)
    for i=1,mb_dim/2 do
        data[i] = state_prime[i]
    end
    target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2)
    target[{{mb_dim/2+1,-1}}] = torch.zeros(mb_dim/2)

    data[{{mb_dim/2+1,-1}}]  = gen_network:forward{state,action}

    output = network:forward{state:cat(state,1),action:cat(action,1),data}
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

    state,action,state_prime = get_data(mb_dim)
    target:zero()
    --target:zero():add(1)
    local output = full_network:forward{state,action}
    local disc_loss = bce_crit:forward(output,target)
    local disc_grad = bce_crit:backward(output,target):clone()
    --full_network:backward(state,disc_grad)
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
        local state,action,state_prime = get_complete_data()
        local output = full_network:forward{state,action}
        gnuplot.figure(plot1)
        gnuplot.imagesc(gen_network.output[{{1,num_state}}]+
                        gen_network.output[{{num_state+1,num_state*2}}]+
                        gen_network.output[{{num_state*2+1,num_state*3}}]+
                        gen_network.output[{{num_state*3+1,-1}}]
                        )
        gnuplot.figure(plot2)
        --[[
        gnuplot.imagesc(state_prime[{{1,num_state}}]+
                        state_prime[{{num_state*1+1,num_state*2}}]+
                        state_prime[{{num_state*2+1,num_state*3}}]+
                        state_prime[{{num_state*3+1,-1}}]
                        )
                        --]]
        gnuplot.bar(network.output)
        print(s,a,sPrime,network.output[1][1])
        print(gen_network.output[num_state])
        print(gen_network.output[num_state*2])
        print(gen_network.output[num_state*3])
        print(gen_network.output[num_state*4])
        net_reward = 0
        cumloss = 0
    end
end

