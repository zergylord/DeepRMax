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
dropout_p = .1
fake_p = .5
--factored = true
--factored_gen = true
--rev_grad = true
gen_hid_dim = 100
noise_dim = 10
if factored then
    --Discrim
    local input = nn.Identity()()
    local action = nn.Identity()()
    print('dropout')
    hid = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(in_dim,hid_dim)(input)))
    factor = nn.Dropout(dropout_p)(nn.CMulTable(){nn.Linear(hid_dim,fact_dim)(hid),nn.Linear(act_dim,fact_dim)(action)})
    --unneeded?
    --hid2 = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(fact_dim,hid_dim)(factor)))
    local out_lin = nn.Linear(fact_dim,out_dim)
    local output = nn.Sigmoid()(out_lin(factor))
    network = nn.gModule({input,action},{output})
else
    --Discrim
    local input = nn.Identity()()
    local action = nn.Identity()()
    print('dropout')
    hid = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(in_dim+act_dim,hid_dim)(nn.JoinTable(2){input,action})))
    hid2 = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid)))
    --one more layer here?
    local out_lin = nn.Linear(hid_dim,out_dim)
    local output = nn.Sigmoid()(out_lin(hid2))
    network = nn.gModule({input,action},{output})
end
if factored_gen then
    --needed?
    --Gen
    local input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
    local factor = nn.CMulTable(){nn.Linear(gen_hid_dim,fact_dim)(hid),nn.Linear(act_dim,fact_dim)(action)}
    local last_hid = nn.ReLU()(nn.Linear(fact_dim,gen_hid_dim)(factor))
    local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(last_hid))
    gen_network = nn.gModule({input,action},{output})
else
    --Gen
    local input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
    local last_hid = nn.ReLU()(nn.Linear(gen_hid_dim,gen_hid_dim)(hid))
    local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(last_hid))
    local action =nn.Sigmoid()( nn.Linear(gen_hid_dim,act_dim)(last_hid))
    gen_network = nn.gModule({input},{output,action})
end
--full
local input = nn.Identity()()
local action = nn.Identity()()
if rev_grad then
    local state,action = gen_network(input):split(2)
    connect= {nn.GradientReversal()(state),nn.GradientReversal()(action)}
else
    connect= gen_network(input)
end
local full_out = network(connect)
full_network = nn.gModule({input},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = 320
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
local get_noise = function(dim)
    local noise = torch.zeros(dim,noise_dim)
    for i=1,dim do
        noise[i][torch.random(noise_dim)] = 1
    end
    return noise
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
local data_a = torch.zeros(mb_dim,act_dim)
num_real = math.floor(mb_dim*(1-fake_p))
num_fake = mb_dim - num_real
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:training()
    
    state,action,_ = get_data(mb_dim)
    for i=1,num_real do
        data[i] = state[i]
        data_a[i] = action[i]
    end
    target[{{1,num_real}}] = torch.ones(num_real)
    target[{{num_real+1,-1}}] = torch.zeros(num_fake)

    data[{{num_real+1,-1}}],data_a[{{num_real+1,-1}}]  = unpack(gen_network:forward(get_noise(num_fake)))

    output = network:forward{data,data_a}
    loss = bce_crit:forward(output,target)
    grad = bce_crit:backward(output,target)
    network:backward({data,data_a},grad)
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
    full_network:backward(state,disc_grad)
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
        local output,action = unpack(gen_network:forward(get_noise(mb_dim)))
        gnuplot.imagesc(output[1]:reshape(28,28))
        print(action[1])
        net_reward = 0
        cumloss = 0
    end
end

