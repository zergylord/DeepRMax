require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
f = hdf5.open('mnist.hdf5')
mnist_data = f:read():all()
mask = mnist_data.t_train:ne(8):reshape(50000,1):expandAs(mnist_data.x_train)
not8 = mnist_data.x_train[mask]
in_dim = mnist_data.x_train:size(2)
not8 = not8:reshape(not8:size(1)/in_dim,in_dim)
notnot8 = mnist_data.x_train[mask:eq(0)]
notnot8 = notnot8:reshape(notnot8:size(1)/in_dim,in_dim)

hid_dim = 500
out_dim = 1
--Discrim
local input = nn.Identity()()
local hid_lin = nn.Linear(in_dim,hid_dim)
local hid = nn.ReLU()(hid_lin(input))
local out_lin = nn.Linear(hid_dim,out_dim)
local output = nn.Sigmoid()(out_lin(hid))
network = nn.gModule({input},{output})
--Gen
noise_dim = 20
gen_hid_dim = 500
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(hid))
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
local mb_dim = 320
local data = torch.zeros(mb_dim,in_dim)
local target = torch.zeros(mb_dim,1)
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:training()
    
    samples = torch.randperm(not8:size(1))
    for i=1,mb_dim/2 do
        data[i] = not8[samples[i]]
    end
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
local refresh = 5e2
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
        print(output[{{1,mb_dim/2}}]:sum(),output[{{mb_dim/2+1,-1}}]:sum())
        net_reward = 0
        cumloss = 0
    end
end

