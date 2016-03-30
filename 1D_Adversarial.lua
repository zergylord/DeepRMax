require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
in_dim = 1
hid_dim = 10
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
gen_hid_dim = 10
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
local mb_dim = 320
local data = torch.zeros(mb_dim,in_dim)
local target = torch.zeros(mb_dim,1)
local mu = torch.Tensor{4}
local sigma = torch.rand(in_dim)
local train_dis = function()
    network:training()

    data[{{1,mb_dim/2}}] = distributions.mvn.rnd(torch.zeros(mb_dim/2,in_dim),mu,sigma) --torch.randn(mb_dim/2,in_dim)*5-1
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
local train_gen = function()
    network:evaluate()

    local noise_data = torch.randn(mb_dim,noise_dim)
    target:zero()
    local output = full_network:forward(noise_data)
    local loss = bce_crit:forward(output,target)
    local grad = bce_crit:backward(output,target)
    full_network:backward(data,grad)
    return loss,dw
end
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
    learningRate  = 1e-5
    }
local num_steps = 1e6
local refresh = 5e2
local cumloss =0 
local plot1 = gnuplot.figure()
local plot2 = gnuplot.figure()
total_data = torch.zeros(mb_dim*(num_steps/refresh),in_dim)
total_output = torch.zeros(mb_dim*(num_steps/refresh),1)
for i=1,num_steps do
    x,batchloss = optim.rmsprop(train,w,config)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        gnuplot.figure(plot2)
        x_ax = torch.Tensor{1,7}
        gnuplot.axis{x_ax[1],x_ax[2],-5,5}
        gnuplot.plot({data[{{mb_dim/2+1,-1}}],'+'},{data[{{1,mb_dim/2}}],'+'})

        data = torch.rand(mb_dim,in_dim):add(-.5):mul(x_ax[2]-x_ax[1]):add(x_ax[1]+(x_ax[2]-x_ax[1])/2)
        data2 = torch.rand(1000,in_dim):add(-.5):mul(10) 
        output = network:forward(data)
        total_data[{{mb_dim*(i/refresh-1)+1,mb_dim*(i/refresh)}}] = data
        total_output[{{mb_dim*(i/refresh-1)+1,mb_dim*(i/refresh)}}] = output
        gnuplot.figure(plot1)
        gnuplot.axis{x_ax[1],x_ax[2],0,1}
        local values = distributions.mvn.pdf(data2,mu,sigma)
        gnuplot.plot({total_data[{{1,mb_dim*(i/refresh)},1}],total_output[{{1,mb_dim*(i/refresh)},1}],'.'},{data2[{{},1}],values[{{},1}],'.'},{x_ax,torch.Tensor{.5,.5}})
        
        net_reward = 0
        cumloss = 0
    end
end

