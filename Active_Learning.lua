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
local last_hid = nn.ReLU()(nn.Linear(gen_hid_dim,gen_hid_dim)(hid))
local output = nn.Linear(gen_hid_dim,in_dim)(last_hid)
gen_network = nn.gModule({input},{output})
--Pessimal Gen
noise_dim = 20
gen_hid_dim = 10
local input = nn.Identity()()
local hid = nn.Tanh()(nn.Linear(noise_dim,gen_hid_dim)(input))
local output = nn.Linear(gen_hid_dim,in_dim)(hid)
pes_gen_network = nn.gModule({input},{output})
pes_w,pes_dw = pes_gen_network:getParameters()
pes_config = {
    learningRate  = 1e-3
    }

--full
local full_input = nn.Identity()()
--local connect= nn.GradientReversal()(gen_network(full_input))
local connect= gen_network(full_input)
local full_out = network(connect)
full_network = nn.gModule({full_input},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = 320
local mb_data = torch.zeros(mb_dim,in_dim)
local target = torch.zeros(mb_dim,1)
local mu = torch.Tensor{4}
local sigma = torch.rand(in_dim)
local data = torch.zeros(1e5,in_dim)
data[{{1,mb_dim/2}}] = distributions.mvn.rnd(torch.zeros(mb_dim/2,in_dim),mu,sigma) --torch.randn(mb_dim/2,in_dim)*5-1
local dind = mb_dim/2
local train_dis = function()
    network:training()

    target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2)
    mb_data[{{1,mb_dim/2}}] = distributions.cat.rnd(mb_dim/2,torch.ones(dind),{categories=data[{{1,dind}}]})

    noise_data = torch.randn(mb_dim/2,noise_dim)
    target[{{mb_dim/2+1,-1}}] = torch.zeros(mb_dim/2)
    mb_data[{{mb_dim/2+1,-1}}]  = gen_network:forward(noise_data)

    output = network:forward(mb_data)
    loss = bce_crit:forward(output,target)
    grad = bce_crit:backward(output,target)
    network:backward(mb_data,grad)
    r = (output[{{1,mb_dim/2}}]:sum() + output[{{mb_dim/2+1,-1}}]:mul(-1):add(1):sum() )/mb_dim
    net_reward = net_reward + r
    return loss,dw
end
local train_gen = function()
    network:evaluate()

    local noise_data = torch.randn(mb_dim,noise_dim)
    target:zero():add(1)
    local output = full_network:forward(noise_data)
    local loss = bce_crit:forward(output,target)
    local grad = bce_crit:backward(output,target)
    full_network:backward(noise_data,grad)
    return loss,dw
end
local train_pes_gen = function(x)
    if x ~= pes_w then
        pes_w:copy(x)
    end
    network:evaluate()
    pes_gen_network:zeroGradParameters()

    local noise_data = torch.randn(mb_dim,noise_dim)
    target:zero()
    local fake = pes_gen_network:forward(noise_data)
    local output = network:forward(fake)
    local loss = bce_crit:forward(output,target)
    local grad = bce_crit:backward(output,target)
    local gen_grad = network:backward(fake,grad)
    pes_gen_network:backward(noise_data,gen_grad)
    return loss,pes_dw
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
    learningRate  = 1e-3
    }
local num_steps = 1e6
local refresh = 1e1
local cumloss =0 
local plot1 = gnuplot.figure()
local plot2 = gnuplot.figure()
total_data = torch.zeros(mb_dim*(num_steps/refresh),in_dim)
total_output = torch.zeros(mb_dim*(num_steps/refresh),1)

log2 = function(x) return torch.log(x)/torch.log(2) end
H = function(p) return log2(p)*(-p)-log2(-p+1)*(-p+1) end
get_adverse_sample = function()
    local sample = data[torch.random(dind)]
    local target = torch.Tensor{.5}
    local output = network:forward(sample)
    local initial = output[1]
    local count = 0
    local limit = 100
    while H(output) > .6 do
        if count > limit then
            return sample
        end
        output = network:forward(sample)
        --loss = bce_crit:forward(output,target)
        --grad = bce_crit:backward(output,target)
        --network:backward(mb_data,grad)
        local grad = -log2(output[1]/(1-output[1]))
        network:backward(mb_data,torch.Tensor{grad})
        sample = sample - network.gradInput:mul(1)
        count = count + 1
    end
    print('yay')
    final = network:forward(sample)[1]
    local x = H(final)
    print(initial,final,x)
    return math.min(10,math.max(-10,sample[1]))
end
mb_H = function(p) return log2(p):cmul(-p)-log2(-p+1):cmul(-p+1) end
get_best_random_sample = function()
    local d = torch.randn(100,1):mul(5)
    local output = network:forward(d)
    local _,ind =  mb_H(output):min(1)
    return d[ind[1][1] ][1]
end
pes_qual = 1
pes_qual_count = 0
get_pes_sample = function()
    local noise_data = torch.randn(1,noise_dim)
    local fake = pes_gen_network:forward(noise_data)
    local output = network:forward(fake)
    pes_qual = (pes_qual*pes_qual_count + (1-H(output[1][1])))/(pes_qual_count+1)
    pes_qual_count = pes_qual_count + 1
    return fake[1]
end



for i=1,num_steps do
        dind = dind + 1
        --data[dind] = get_adverse_sample()
        --data[dind] = get_best_random_sample()
        data[dind] = get_pes_sample()
        --data[dind] = torch.randn(1):add(4):mul(10)
        _,batchloss = optim.adam(train,w,config)
        for iter = 1,5 do
            optim.adam(train_pes_gen,pes_w,config_pes)
        end
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,pes_qual,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        x_ax = torch.Tensor{1,7}
        gnuplot.figure(2)
        gnuplot.axis{-30,30,'',''}
        gnuplot.title('gen data')
        gnuplot.hist(gen_network:forward(torch.randn(dind,noise_dim)))
        gnuplot.figure(4)
        gnuplot.axis{-30,30,'',''}
        gnuplot.title('pes data')
        gnuplot.hist(pes_gen_network:forward(torch.randn(dind,noise_dim)))
        gnuplot.figure(3)
        gnuplot.axis{-30,30,'',''}
        gnuplot.title('real data')
        --gnuplot.hist(data[{{dind-refresh+1,dind}}])
        gnuplot.hist(data[{{1,dind}}])

        data1 = torch.rand(mb_dim,in_dim):add(-.5):mul(x_ax[2]-x_ax[1]):add(x_ax[1]+(x_ax[2]-x_ax[1])/2)
        data2 = torch.rand(1000,in_dim):add(-.5):mul(10) 
        output = network:forward(data1)
        total_data[{{mb_dim*(i/refresh-1)+1,mb_dim*(i/refresh)}}] = data1
        total_output[{{mb_dim*(i/refresh-1)+1,mb_dim*(i/refresh)}}] = output
        gnuplot.figure(1)
        gnuplot.axis{x_ax[1],x_ax[2],0,1}
        local values = distributions.mvn.pdf(data2,mu,sigma)
        gnuplot.plot({total_data[{{1,mb_dim*(i/refresh)},1}],total_output[{{1,mb_dim*(i/refresh)},1}],'.'},{data2[{{},1}],values[{{},1}],'.'},{x_ax,torch.Tensor{.5,.5}})
        
        net_reward = 0
        cumloss = 0
    end
end

