use_gpu = true
--rev_grad = true
use_action = true
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
    act_dim = 4
    in_dim = in_dim + act_dim
    act_dict = torch.eye(act_dim):float()
    if use_gpu then
        act_dict = act_dict:cuda()
    end
end
if use_gpu then
    notnot8 = notnot8:cuda()
    not8 = not8:cuda()
end
hid_dim = 120
noise_dim = 10
gen_hid_dim = 1200
dropout = .1
mb_dim = 100
out_dim = 1
if use_gpu then
    --Discrim
    local input = nn.Identity():cuda()()
    local hid_lin = nn.Linear(in_dim,hid_dim):cuda()
    local hid = nn.Dropout(dropout):cuda()(nn.ReLU():cuda()(hid_lin(input)))
    local out_lin = nn.Linear(hid_dim,out_dim):cuda()
    local output = (nn.Sigmoid():cuda()(out_lin(hid)))
    network = nn.gModule({input},{output})
    --Gen
    local input = nn.Identity():cuda()()
    local hid = nn.Dropout(dropout):cuda()(nn.ReLU():cuda()(nn.Linear(noise_dim,gen_hid_dim):cuda()(input)))
    local output =nn.Sigmoid():cuda()( nn.Linear(gen_hid_dim,in_dim):cuda()(hid))
    gen_network = nn.gModule({input},{output})
else
    --Discrim
    local input = nn.Identity()()
    local hid_lin = nn.Linear(in_dim,hid_dim)
    local hid = nn.ReLU()(hid_lin(input))
    local out_lin = nn.Linear(hid_dim,out_dim)
    local output = nn.Sigmoid()(out_lin(hid))
    network = nn.gModule({input},{output})
    --Gen
    local input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
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
    target = torch.zeros(mb_dim,1):cuda()
else
    data = torch.zeros(mb_dim,in_dim)
    target = torch.zeros(mb_dim,1)
end
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)
local get_noise = function(dim)
    --one-hot noise
    local noise = torch.zeros(dim,noise_dim)
    for i=1,dim do
        noise[i][torch.random(noise_dim)] = 1
    end
    --just gaussian
    return torch.randn(dim,noise_dim)
end
local train_dis = function(x)
    gen_network:evaluate()
    samples = torch.randperm(not8:size(1))
    for i=1,mb_dim/2 do
        if use_action then
            data[i] = not8[samples[i]]:cat(act_dict[torch.random(act_dim)])
        else
            data[i] = not8[samples[i]]
        end
    end
    if use_gpu then
        target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2):cuda()
        noise_data = get_noise(mb_dim/2):cuda()
    else
        target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2)
        noise_data = get_noise(mb_dim/2)
    end

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
    network:evaluate()
    local noise_data
    if use_gpu then
        noise_data = torch.randn(mb_dim,noise_dim):cuda()
    else
        noise_data = torch.randn(mb_dim,noise_dim)
    end
    if rev_grad then
        target:zero()
    else
        target:zero():add(1)
    end
    local output = full_network:forward(noise_data)
    local loss = bce_crit:forward(output,target)
    local grad = bce_crit:backward(output,target)
    full_network:backward(noise_data,grad)
    return loss,dw
end
local train = function(x)
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

config_dis = {
    learningRate  = 1e-5
    }
config_gen = {
    learningRate  = 1e-5
    }
local num_steps = 1e6
local refresh = 5e3
local cumloss =0 
local plot1 = gnuplot.figure()
local plot2 = gnuplot.figure()
local plot3 = gnuplot.figure()
for i=1,num_steps do
    --[[
    for k=1,1 do
        x,batchloss = optim.adam(train_dis,w,config_dis)
    end
    x,batchloss = optim.adam(train_gen,w,config_gen)
    --]]
    x,batchloss = optim.adam(train,w,config_gen)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        output = network:forward(data)
        timer:reset()
        gnuplot.figure(plot1)
        if use_action then
            gnuplot.imagesc(data[{{mb_dim/2+1},{1,in_dim-act_dim}}]:reshape(28,28))
        else
            gnuplot.imagesc(data[{{mb_dim/2+1}}]:reshape(28,28))
        end
        
        
        gnuplot.figure(plot2)
        samples1 = torch.randperm(not8:size(1))
        samples2 = torch.randperm(notnot8:size(1))
        gnuplot.bar(network.output)
        gnuplot.axis{0,100,0,1}


        gnuplot.figure(plot3)
        if use_action then
            --show all actions (bottom half generated)
            gnuplot.imagesc(data[{{},{in_dim-act_dim+1,-1}}])
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

