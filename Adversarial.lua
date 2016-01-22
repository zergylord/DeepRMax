require 'nngraph'
require 'optim'
in_dim = 5
hid_dim = 3
out_dim = 1
--Discrim
local input = nn.Identity()()
local hid_lin = nn.Linear(in_dim,hid_dim)
local hid = nn.ReLU()(hid_lin(input))
local out_lin = nn.Linear(hid_dim,out_dim)
local output = nn.Sigmoid()(out_lin(hid))
network = nn.gModule({input},{output})
--Gen
noise_dim = 2
gen_hid_dim = 3
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
local output = nn.Linear(gen_hid_dim,in_dim)(hid)
gen_network = nn.gModule({input},{output})

--full
local full_input = nn.Identity()()
local connect= nn.GradientReversal()(gen_network(full_input))
local full_out = network(connect)
full_network = nn.gModule({full_input},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = 32
local data = torch.zeros(mb_dim,in_dim)
local target = torch.zeros(mb_dim,1)
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    network:zeroGradParameters()
    network:training()

    data[{{1,mb_dim/2}}] = torch.randn(mb_dim/2,in_dim)*5-1
    target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2)

    noise_data = torch.randn(mb_dim/2,noise_dim)
    target[{{mb_dim/2+1,-1}}] = torch.zeros(mb_dim/2)
    data[{{mb_dim/2+1,-1}}]  = gen_network:forward(noise_data)

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

    local noise_data = torch.randn(mb_dim,noise_dim)
    target:zero()
    local output = full_network:forward(noise_data)
    local loss = bce_crit:forward(output,target)
    local grad = bce_crit:backward(output,target)
    full_network:backward(data,grad)
    return loss,dw
end
config = {
    learningRate  = 1e-3
    }
local num_steps = 1e6
local refresh = 1e2
local cumloss =0 
for i=1,num_steps do
    x,batchloss = optim.rmsprop(train_dis,w,config)
    x,batchloss = optim.rmsprop(train_gen,w,config)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        net_reward = 0
        cumloss = 0
    end
end

