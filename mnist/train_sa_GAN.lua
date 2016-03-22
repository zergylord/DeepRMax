use_gpu = true
--rev_grad = true
--use_action = true
--noise_mag = .25
act_dim = 4
if use_gpu then
    require 'cunn'
end
require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
torch.setnumthreads(1)
f = hdf5.open('mnist.hdf5')
mnist_data = f:read():all()
mask = mnist_data.t_train:ne(8):reshape(50000,1):expandAs(mnist_data.x_train)
not8 = mnist_data.x_train[mask]
in_dim = mnist_data.x_train:size(2)
not8 = not8:reshape(not8:size(1)/in_dim,in_dim)
notnot8 = mnist_data.x_train[mask:eq(0)]
notnot8 = notnot8:reshape(notnot8:size(1)/in_dim,in_dim)
if use_action then
    act_dict = torch.eye(act_dim):float()
    if use_gpu then
        act_dict = act_dict:cuda()
    end
end
if use_gpu then
    notnot8 = notnot8:cuda()
    not8 = not8:cuda()
end
hid_dim = 1000
noise_dim = 20
gen_hid_dim = 1000
dropout = 0.5
mb_dim = 320
out_dim = 1
if use_gpu then
    --Discrim
    local input = nn.Identity():cuda()()
    local action = nn.Identity():cuda()()
    --local action_hid = nn.ReLU():cuda()(nn.Linear(act_dim,in_dim):cuda()(action))
    if use_action then
        hid_lin = nn.Linear(in_dim+act_dim,hid_dim):cuda()
        hid = nn.Dropout(dropout):cuda()(nn.ReLU():cuda()(hid_lin(nn.JoinTable(2):cuda(){input,action})))
    else
        hid_lin = nn.Linear(in_dim,hid_dim):cuda()
        hid = nn.Dropout(dropout):cuda()(nn.ReLU():cuda()(hid_lin(input)))
    end
    local out_lin = nn.Linear(hid_dim,out_dim):cuda()
    local output = (nn.Sigmoid():cuda()(out_lin(hid)))
    if use_action then
        network = nn.gModule({input,action},{output})
    else
        network = nn.gModule({input},{output})
    end
    --Gen
    local input = nn.Identity():cuda()()
    local hid = nn.BatchNormalization(gen_hid_dim):cuda()(nn.ReLU():cuda()(nn.Linear(noise_dim,gen_hid_dim):cuda()(input)))
    local output =nn.Sigmoid():cuda()( nn.Linear(gen_hid_dim,in_dim):cuda()(hid))
    if use_action then
        local action =nn.Sigmoid():cuda()( nn.Linear(gen_hid_dim,act_dim):cuda()(hid))
        gen_network = nn.gModule({input},{output,action})
    else
        gen_network = nn.gModule({input},{output})
    end
else
    --Discrim
    local input = nn.Identity()()
    local hid_lin = nn.Linear(in_dim,hid_dim)
    local hid = nn.Dropout(dropout)(nn.ReLU()(hid_lin(input)))
    local out_lin = nn.Linear(hid_dim,out_dim)
    local output = (nn.Sigmoid()(out_lin(hid)))
    network = nn.gModule({input},{output})
    --Gen
    local input = nn.Identity()()
    local hid = nn.BatchNormalization(gen_hid_dim)(nn.Dropout(dropout)(nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))))
    local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(hid))
    gen_network = nn.gModule({input},{output})
end
--full
local full_input = nn.Identity()()
if rev_grad then
    connect= nn.GradientReversal():cuda()(gen_network(full_input))
else
    connect= gen_network(full_input)
end
local full_out = network(connect)
full_network = nn.gModule({full_input},{full_out})


w,dw = full_network:getParameters()
print(w:type())
local timer = torch.Timer()
if use_gpu then
    bce_crit = nn.BCECriterion():cuda()
else
    bce_crit = nn.BCECriterion()
end
local net_reward = 0
if use_gpu then
    data = torch.zeros(mb_dim,in_dim):cuda()
    action_data = torch.zeros(mb_dim,act_dim):cuda()
    dis_target = torch.zeros(mb_dim,1):cuda()
    dis_target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2):cuda()
    if rev_grad then
        gen_target = torch.zeros(mb_dim,1):cuda()
    else
        gen_target = torch.ones(mb_dim,1):cuda()
    end
else
    data = torch.zeros(mb_dim,in_dim)
    dis_target = torch.zeros(mb_dim,1)
    dis_target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2)
    if rev_grad then
        gen_target = torch.zeros(mb_dim,1)
    else
        gen_target = torch.ones(mb_dim,1)
    end
end


local get_noise = function(dim)
    --one-hot noise
    local noise = torch.zeros(dim,noise_dim)
    for i=1,dim do
        noise[i][torch.random(noise_dim)] = 1
    end
    --just gaussian
    return torch.randn(dim,noise_dim)
end
--[[ takes a view of the memory 
    and fills it with data
--]]
local get_data = function(data,action_data)
    local samples = torch.randperm(not8:size(1))
    --local data = torch.zeros(num,dim)
    num = data:size(1)
    dim = data:size(2)
    for i=1,num do
        if use_action then
            data[i] = not8[samples[i] ]
            --action_data[i] = act_dict[torch.random(act_dim)]
            action_data[i] = torch.rand(act_dim):mul(noise_mag)
            action_data[i][torch.random(act_dim)] = 1 - torch.rand(1):mul(noise_mag)[1]
        else
            data[i] = not8[samples[i] ]
        end
    end
end

local train_dis = function()
    --gen_network:evaluate()
    network:training()
    if use_action then
        data_func(data[{{1,mb_dim/2}}],action_data[{{1,mb_dim/2}}])
    else
        data_func(data[{{1,mb_dim/2}}])
    end

    local noise_data
    if use_gpu then
        noise_data = get_noise(mb_dim/2):cuda()
    else
        noise_data = get_noise(mb_dim/2)
    end
    local output
    if use_action then
        data[{{mb_dim/2+1,-1}}],action_data[{{mb_dim/2+1,-1}}]  = unpack(gen_network:forward(noise_data))
        output = network:forward{data,action_data}
    else
        data[{{mb_dim/2+1,-1}}]  = gen_network:forward(noise_data)
        output = network:forward(data)
    end

    local loss = bce_crit:forward(output,dis_target)
    local grad = bce_crit:backward(output,dis_target)
    if use_action then
        network:backward({data,action_data},grad)
    else
        network:backward(data,grad)
    end
    r = (output[{{1,mb_dim/2}}]:sum() + output[{{mb_dim/2+1,-1}}]:mul(-1):add(1):sum() )/mb_dim
    net_reward = net_reward + r
    return loss,dw
end
local train_gen = function()
    network:evaluate()
    --network:training()
    local noise_data
    if use_gpu then
        noise_data = get_noise(mb_dim):cuda()
    else
        noise_data = get_noise(mb_dim)
    end
    local output = full_network:forward(noise_data)
    local loss = bce_crit:forward(output,gen_target)
    local grad = bce_crit:backward(output,gen_target)
    full_network:backward(noise_data,grad)
    return loss,dw
end
--[[
--Public
--sets the function called in train_dis to
--generate data
--]]
set_data_func = function(func)
    data_func = func
end
--[[
--Public
--single training step
--assumes get_data has been set to a minibatch generating function
--]]
train = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    local loss = 0
    loss = loss + train_gen()
    network:zeroGradParameters()
    --sees 3x fake examples, if not zerod
    for i = 1,1 do
        loss = loss + train_dis()
    end
    return loss,dw
end

config = {
    learningRate  = 1e-3
    }
standard_training = function()
local num_steps = 1e6
local refresh = 1e3
local cumloss =0 
local plot1 = gnuplot.figure()
local plot2 = gnuplot.figure()
local plot3 = gnuplot.figure()
set_data_func(get_data)
for i=1,num_steps do
    x,batchloss = optim.adam(train,w,config)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        if use_action then
            output = network:forward{data,action_data}
        else
            output = network:forward(data)
        end
        timer:reset()
        gnuplot.figure(plot1)
        gnuplot.imagesc(data[{{mb_dim/2+1}}]:reshape(28,28))
        
        
        gnuplot.figure(plot2)
        samples1 = torch.randperm(not8:size(1))
        samples2 = torch.randperm(notnot8:size(1))
        gnuplot.bar(network.output)
        gnuplot.axis{0,mb_dim,0,1}


        gnuplot.figure(plot3)
        if use_action then
            --show all actions (bottom half generated)
            gnuplot.imagesc(action_data)
            --test unused actions
        else
            for i=1,mb_dim do
                if i<=mb_dim/2 then
                    data[i] = not8[samples1[i]]
                else
                    data[i] = notnot8[samples2[i]]
                end 
            end
            output = network:forward(data)
            print(output[{{1,mb_dim/2}}]:mean(),output[{{mb_dim/2+1,-1}}]:mean())
        end
        net_reward = 0
        cumloss = 0
    end
end
end
