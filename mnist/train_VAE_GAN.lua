use_gpu = true
act_dim = 4
if use_gpu then
    require 'cunn'
end
require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
require 'cuReparametrize'
require 'KLDCriterion'
torch.setnumthreads(1)
--use_mnist = true
if use_mnist then
    f = hdf5.open('mnist.hdf5')
    mnist_data = f:read():all()
    mask = mnist_data.t_train:ne(8):reshape(50000,1):expandAs(mnist_data.x_train)
    not8 = mnist_data.x_train[mask]
    in_dim = mnist_data.x_train:size(2)
    not8 = not8:reshape(not8:size(1)/in_dim,in_dim)
    notnot8 = mnist_data.x_train[mask:eq(0)]
    notnot8 = notnot8:reshape(notnot8:size(1)/in_dim,in_dim)
    act_dict = torch.eye(act_dim):float()
    if use_gpu then
        act_dict = act_dict:cuda()
        notnot8 = notnot8:cuda()
        not8 = not8:cuda()
    end
else
    in_dim = num_state or 10
end
hid_dim = 1000
gen_hid_dim = 1000
dropout = 0.5
mb_dim = 320
out_dim = 1
fact_dim = 10
--Discrim
local input = nn.Identity():cuda()()
local action = nn.Identity():cuda()()
hid_lin = nn.Linear(in_dim,hid_dim):cuda()
hid = nn.Dropout(dropout):cuda()(nn.ReLU():cuda()(hid_lin(input)))
local factor = nn.Dropout(dropout):cuda()(nn.CMulTable():cuda(){nn.Linear(hid_dim,fact_dim):cuda()(hid),nn.Linear(act_dim,fact_dim):cuda()(action)})
local out_lin = nn.Linear(fact_dim,out_dim):cuda()
local output = (nn.Sigmoid():cuda()(out_lin(factor)))
network = nn.gModule({input,action},{output})
--encoder
local action = nn.Identity():cuda()()
local hid = nn.ReLU():cuda()(nn.Linear(act_dim,hid_dim):cuda()(action))
noise_dim = 4
local mu = nn.Linear(hid_dim,noise_dim):cuda()(hid)
local sigma = nn.Linear(hid_dim,noise_dim):cuda()(hid)
encoder = nn.gModule({action},{mu,sigma})
--sampler
local mu = nn.Identity():cuda()()
local sigma = nn.Identity():cuda()()
local sample = nn.Reparametrize(noise_dim){mu,sigma}
sampler = nn.gModule({mu,sigma},{sample})

--Gen / decoder
local input = nn.Identity():cuda()()
local noise = nn.Identity():cuda()()
local hid = nn.BatchNormalization(gen_hid_dim):cuda()(nn.ReLU():cuda()(nn.CMulTable():cuda(){nn.Linear(in_dim,gen_hid_dim):cuda()(input),nn.Linear(noise_dim,gen_hid_dim):cuda()(noise)}))
local last_hid = hid
local output =nn.Sigmoid():cuda()( nn.Linear(gen_hid_dim,act_dim):cuda()(last_hid))
gen_network = nn.gModule({input,noise},{output})
--full
local state = nn.Identity()()
local action = nn.Identity()()
connect = gen_network{state,sampler(encoder(action))}
local full_out = network{state,connect}
full_network = nn.gModule({state,action},{full_out})


w,dw = full_network:getParameters()
print(w:type())
local timer = torch.Timer()
if use_gpu then
    bce_crit = nn.BCECriterion():cuda()
    kl_crit = nn.KLDCriterion():cuda()
else
    bce_crit = nn.BCECriterion()
end
local net_reward = 0
data = torch.zeros(mb_dim,in_dim):cuda()
action_data = torch.zeros(mb_dim,act_dim):cuda()
dis_target = torch.zeros(mb_dim,1):cuda()
dis_target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2):cuda()
if rev_grad then
    gen_target = torch.zeros(mb_dim,1):cuda()
else
    gen_target = torch.ones(mb_dim,1):cuda()
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
        data[i] = not8[samples[i] ]
        --action_data[i] = act_dict[torch.random(act_dim)]
        action_data[i] = torch.rand(act_dim):mul(noise_mag)
        action_data[i][torch.random(act_dim)] = 1 - torch.rand(1):mul(noise_mag)[1]
    end
end
get_noise = function(dim)
    return torch.randn(dim,noise_dim):cuda()
end

local train_dis = function()
    --gen_network:evaluate()
    network:training()
    data_func(data[{{1,mb_dim}}],action_data[{{1,mb_dim}}])

    local output
    --z = sampler:forward(encoder:forward(action_data[{{mb_dim/2+1,-1}}]))
    z = get_noise(mb_dim/2)
    action_data[{{mb_dim/2+1,-1}}]  = gen_network:forward{data[{{mb_dim/2+1,-1}}],z}
    output = network:forward{data,action_data}
    last_compare = output

    local loss = bce_crit:forward(output,dis_target)
    local grad = bce_crit:backward(output,dis_target)
    network:backward({data,action_data},grad)
    r = (output[{{1,mb_dim/2}}]:sum() + output[{{mb_dim/2+1,-1}}]:mul(-1):add(1):sum() )/mb_dim
    net_reward = net_reward + r
    return loss,dw
end
local train_gen = function()
    network:evaluate()
    --network:training()
    data_func(data[{{1,mb_dim}}],action_data[{{1,mb_dim}}])
    local output = full_network:forward{data,action_data}
    local loss = bce_crit:forward(output,gen_target)
    local grad = bce_crit:backward(output,gen_target)
    --full_network:backward({data,action_data},grad)

    --
    grad = network:backward({data,gen_network.output},grad)[2]
    loss = loss + bce_crit:forward(gen_network.output,action_data)
    grad = grad + bce_crit:backward(gen_network.output,action_data)
    --grad = bce_crit:backward(gen_network.output,action_data)
    local sampler_grad = gen_network:backward({data,action_data},grad)[2]
    local enc_grad = sampler:backward(encoder.output,sampler_grad)
    loss = loss + kl_crit:forward(encoder.output)
    kl_grad = kl_crit:backward(encoder.output)
    enc_grad[1] = enc_grad[1] + kl_grad[1]
    enc_grad[2] = enc_grad[2] + kl_grad[2]
    encoder:backward(action_data,enc_grad)
    --]]
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
--return yes/no and value for storage
get_knownness = function(output,ind)
    local chance_unknown = (1 - H(output[ind][1]))^(1/temp)
    --nan check
    if chance_unknown ~= chance_unknown then
        chance_unknown = 1
    end
    return chance_unknown > torch.rand(1)[1], chance_unknown
end

config = {
    learningRate  = 1e-3
    }
standard_training = function()
 noise_mag = .15
local num_steps = 1e6
local refresh = 1e2
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
        output = network:forward{data,action_data}
        timer:reset()
        gnuplot.figure(plot1)
        gnuplot.imagesc(action_data)
        
        
        gnuplot.figure(plot2)
        samples1 = torch.randperm(not8:size(1))
        samples2 = torch.randperm(notnot8:size(1))
        gnuplot.bar(network.output)
        gnuplot.axis{0,mb_dim,0,1}


        gnuplot.figure(plot3)
        data_func(data[{{1,mb_dim}}],action_data[{{1,mb_dim}}])
        for i=1,mb_dim do
            if i<=mb_dim/2 then
                data[i] = not8[samples1[i]]
            else
                data[i] = notnot8[samples2[i]]
            end 
        end
        output = network:forward{data,action_data}
        gnuplot.bar(network.output)
        gnuplot.axis{0,mb_dim,0,1}
        print(output[{{1,mb_dim/2}}]:mean(),output[{{mb_dim/2+1,-1}}]:mean())
        net_reward = 0
        cumloss = 0
    end
end
end
