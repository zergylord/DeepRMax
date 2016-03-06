require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
--torch.setnumthreads(1)

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
out_dim = 1
dropout = true
dropout_p = .1
fake_p = .5
rev_grad = true
gen_hid_dim = 100
--Discrim
local input = nn.Identity()()
local action = nn.Identity()()
if dropout then
    print('dropout')
    hid = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(in_dim+act_dim,hid_dim)(nn.JoinTable(2){input,action})))
    hid2 = nn.Dropout(dropout_p)(nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid)))
else
    print('no dropout')
    hid = (nn.ReLU()(nn.Linear(in_dim+act_dim,hid_dim)(nn.JoinTable(2){input,action})))
    hid2 = (nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid)))
end
--one more layer here?
local out_lin = nn.Linear(hid_dim,out_dim)
local output = nn.Sigmoid()(out_lin(hid2))
network = nn.gModule({input,action,sPrime},{output})
--Gen
noise_dim = 40
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
local last_hid = nn.ReLU()(nn.Linear(gen_hid_dim,gen_hid_dim)(hid))
local output =nn.Sigmoid()(nn.Linear(gen_hid_dim,in_dim)(last_hid))
local action = nn.Sigmoid()(nn.Linear(gen_hid_dim,act_dim)(last_hid))
gen_network = nn.gModule({input},{output,action})
--full
local input = nn.Identity()()
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
local mb_dim = 32
local target = torch.zeros(mb_dim,1)
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)

local s,a,sPrime
local get_data = function(dim)
    local state = torch.rand(dim,in_dim):mul(.1)
    local action = torch.rand(dim,act_dim):mul(.1)
    local state_prime = torch.rand(dim,in_dim):mul(.1)

    for i=1,dim do
        s = torch.random(num_state)
        state[i][s] = 1
        a = torch.random(act_dim)
        action[i][a] = 1
        sPrime = T[s][a]
        state_prime[i][sPrime] = 1
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
    local state = torch.rand(dim,in_dim):mul(.1)
    local action = torch.rand(dim,act_dim):mul(.1)
    local state_prime = torch.rand(dim,in_dim):mul(.1)

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
local data_a = torch.zeros(mb_dim,act_dim)
num_real = math.floor(mb_dim*(1-fake_p))
num_fake = mb_dim - num_real
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:training()
    
    state,action,_ = get_data(num_real)
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
    full_network:backward(data,disc_grad)
    recon_loss = 0
    --[[
    local recon_loss = bce_crit:forward(gen_network.output,state_prime)
    local recon_grad = bce_crit:backward(gen_network.output,state_prime):clone()
    --gen_network:backward({state,action},recon_grad:mul(1))
    --]]
    return disc_loss+recon_loss,dw
end
config = {
    learningRate  = 1e-5
    }
local num_steps = 1e6
local refresh = 1e3
local cumloss =0 
local plot1 = gnuplot.figure()
local plot2 = gnuplot.figure()
for i=1,num_steps do
    for k=1,5 do
        x,batchloss = optim.adam(train_dis,w,config)
    end
    x,batchloss = optim.adam(train_gen,w,config)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        local samples = 1
        local y = torch.zeros(40*samples)
        local x = torch.range(1,10):repeatTensor(1,4*samples)[1]
        for i=1,samples do
            local state,action,_ = get_complete_data()
            local output = network:forward{state,action}
            y[{{40*(i-1)+1,40*i}}] = output:clone()
        end
        gnuplot.figure(plot1)
        --print(unpack(gen_network:forward(torch.randn(noise_dim))))
        gnuplot.hist(full_network:forward(get_noise(mb_dim)))
        print(gen_network.output[1][1],gen_network.output[2][1])
        gnuplot.axis{0,1,0,20}
        --[[
        gnuplot.imagesc(gen_network.output[{{1,num_state}}]+
                        gen_network.output[{{num_state+1,num_state*2}}]+
                        gen_network.output[{{num_state*2+1,num_state*3}}]+
                        gen_network.output[{{num_state*3+1,-1}}]
                        )
                        --]]
        gnuplot.figure(plot2)
        --[[
        gnuplot.imagesc(state_prime[{{1,num_state}}]+
                        state_prime[{{num_state*1+1,num_state*2}}]+
                        state_prime[{{num_state*2+1,num_state*3}}]+
                        state_prime[{{num_state*3+1,-1}}]
                        )
                        --]]
        --gnuplot.plot(x,y,'+')
        --gnuplot.hist(y)
        --gnuplot.axis{0,1,0,20}
        gnuplot.bar(gen_network.output[1][1]:cat(gen_network.output[2][1]))
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

