require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
--torch.setnumthreads(1)

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
fact_dim  = 10
out_dim = 1
dropout = true
dropout_p = .1
fake_p = .5
factored = true
factored_gen = true
--rev_grad = true
gen_hid_dim = 100
if factored then
    --Discrim
    local input = nn.Identity()()
    local action = nn.Identity()()
    local sPrime = nn.Identity()()
    if dropout then
        print('dropout')
        hid = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(in_dim,hid_dim)(input)))
        factor = nn.Dropout(dropout_p)(nn.CMulTable(){nn.Linear(hid_dim,fact_dim)(hid),nn.Linear(act_dim,fact_dim)(action)})
        hid2 = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(fact_dim,hid_dim)(factor)))
        factor2 = nn.Dropout(dropout_p)(nn.CMulTable(){nn.Linear(hid_dim,fact_dim)(hid2),nn.Linear(in_dim,fact_dim)(sPrime)})
    else
        print('no dropout')
        hid = (nn.ReLU()(nn.Linear(in_dim,hid_dim)(input)))
        factor = (nn.CMulTable(){nn.Linear(hid_dim,fact_dim)(hid),nn.Linear(act_dim,fact_dim)(action)})
        hid2 = (nn.ReLU()(nn.Linear(fact_dim,hid_dim)(factor)))
        factor2 = (nn.CMulTable(){nn.Linear(hid_dim,fact_dim)(hid2),nn.Linear(in_dim,fact_dim)(sPrime)})
    end
    --one more layer here?
    local out_lin = nn.Linear(fact_dim,out_dim)
    local output = nn.Sigmoid()(out_lin(factor2))
    network = nn.gModule({input,action,sPrime},{output})
else
    --Discrim
    local input = nn.Identity()()
    local action = nn.Identity()()
    local sPrime = nn.Identity()()
    if dropout then
        print('dropout')
        hid = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(in_dim+act_dim+in_dim,hid_dim)(nn.JoinTable(2){input,action,sPrime})))
        hid2 = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid)))
    else
        print('no dropout')
        hid = (nn.ReLU()(nn.Linear(in_dim+act_dim+in_dim,hid_dim)(nn.JoinTable(2){input,action,sPrime})))
        hid2 = (nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid)))
    end
    --one more layer here?
    local out_lin = nn.Linear(hid_dim,out_dim)
    local output = nn.Sigmoid()(out_lin(hid2))
    network = nn.gModule({input,action,sPrime},{output})
end
if factored_gen then
    --Gen
    local input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(in_dim,gen_hid_dim)(input))
    local action = nn.Identity()()
    local factor = nn.CMulTable(){nn.Linear(gen_hid_dim,fact_dim)(hid),nn.Linear(act_dim,fact_dim)(action)}
    local last_hid = nn.ReLU()(nn.Linear(fact_dim,gen_hid_dim)(factor))
    local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(last_hid))
    gen_network = nn.gModule({input,action},{output})
else
    --Gen
    local input = nn.Identity()()
    local action = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(in_dim+act_dim,gen_hid_dim)(nn.JoinTable(2){input,action}))
    local last_hid = nn.ReLU()(nn.Linear(gen_hid_dim,gen_hid_dim)(hid))
    local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(last_hid))
    gen_network = nn.gModule({input,action},{output})
end
--full
local input = nn.Identity()()
local action = nn.Identity()()
if rev_grad then
    connect= nn.GradientReversal()(gen_network{input,action})
else
    connect= gen_network{input,action}
end
local full_out = network{input,action,connect}
full_network = nn.gModule({input,action},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = 32
local target = torch.zeros(mb_dim,1)
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)

local s,a,sPrime
local noise_level = .45
local get_data = function(dim)
    local state = torch.Tensor(dim,in_dim)
    local action = torch.zeros(dim,act_dim)
    local state_prime = torch.Tensor(dim,in_dim)
    local shuffle = {}
    for i=1,num_state do
        shuffle[i] = torch.randperm(digit[i]:size(1))
    end

    for i=1,dim do
        s = torch.random(num_state)
        state[i] = digit[s][shuffle[s][i] ]
        a = torch.random(act_dim)
        action[i][a] = 1
        sPrime = T[s][a]
        state_prime[i] = digit[sPrime][shuffle[sPrime][i] ]
    end
    return state,action,state_prime
end
local get_complete_data = function()
    local dim = num_state*act_dim
    local state = torch.Tensor(dim,in_dim)
    local action = torch.zeros(dim,act_dim)
    local state_prime = torch.Tensor(dim,in_dim)
    local shuffle = {}
    for i=1,num_state do
        shuffle[i] = torch.randperm(digit[i]:size(1))
    end

    for a=1,act_dim do
        for s=1,num_state do
            local i = num_state*(a-1)+s
            state[i] = digit[s][shuffle[s][i] ]
            action[i][a] = 1
            sPrime = T[s][a]
            state_prime[i] = digit[sPrime][shuffle[sPrime][i] ]
        end
    end
    return state,action,state_prime
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
    
    state,action,state_prime = get_data(mb_dim)
    for i=1,num_real do
        data[i] = state_prime[i]
    end
    target[{{1,num_real}}] = torch.ones(num_real)
    target[{{num_real+1,-1}}] = torch.zeros(num_fake)

    data[{{num_real+1,-1}}]  = gen_network:forward{state[{{num_real+1,-1}}],action[{{num_real+1,-1}}]}

    output = network:forward{state,action,data}
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

    state,action,state_prime = get_data(mb_dim)
    if rev_grad then
        target:zero()
    else
        target:zero():add(1)
    end
    local output = full_network:forward{state,action}
    local disc_loss = bce_crit:forward(output,target)
    local disc_grad = bce_crit:backward(output,target):clone()
    full_network:backward(state,disc_grad)
    local recon_loss = bce_crit:forward(gen_network.output,state_prime)
    local recon_grad = bce_crit:backward(gen_network.output,state_prime):clone()
    --gen_network:backward({state,action},recon_grad:mul(1))
    return disc_loss+recon_loss,dw
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
local plot3 = gnuplot.figure()
local plot4 = gnuplot.figure()
local plot5 = gnuplot.figure()
for i=1,num_steps do
    for k=1,5 do
        x,batchloss = optim.adam(train_dis,w,config_dis)
    end
    x,batchloss = optim.adam(train_gen,w,config_gen)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        local samples = 1
        local y = torch.zeros(40*samples)
        local x = torch.range(1,10):repeatTensor(1,4*samples)[1]
        for i=1,samples do
            local state,action,state_prime = get_complete_data()
            local output = full_network:forward{state,action}
            y[{{40*(i-1)+1,40*i}}] = output:clone()
        end
        gnuplot.figure(plot1)
        --[[
        gnuplot.imagesc(gen_network.output[{{1,num_state}}]+
                        gen_network.output[{{num_state+1,num_state*2}}]+
                        gen_network.output[{{num_state*2+1,num_state*3}}]+
                        gen_network.output[{{num_state*3+1,-1}}]
                        )
                        --]]
        gnuplot.imagesc(gen_network.output[1]:reshape(28,28))
        gnuplot.figure(plot3)
        gnuplot.imagesc(gen_network.output[1+num_state]:reshape(28,28))
        gnuplot.figure(plot4)
        gnuplot.imagesc(gen_network.output[1+num_state*2]:reshape(28,28))
        gnuplot.figure(plot5)
        gnuplot.imagesc(gen_network.output[1+num_state*3]:reshape(28,28))
        gnuplot.figure(plot2)
        --[[
        gnuplot.imagesc(state_prime[{{1,num_state}}]+
                        state_prime[{{num_state*1+1,num_state*2}}]+
                        state_prime[{{num_state*2+1,num_state*3}}]+
                        state_prime[{{num_state*3+1,-1}}]
                        )
                        --]]
        gnuplot.plot(x,y,'+')
        --gnuplot.bar(network.output)
        gnuplot.plotflush()
        --[[
        print(s,a,sPrime,network.output[1][1])
        print(gen_network.output[num_state])
        print(gen_network.output[num_state*2])
        print(gen_network.output[num_state*3])
        print(gen_network.output[num_state*4])
        --]]
        net_reward = 0
        cumloss = 0
    end
end

