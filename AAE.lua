require 'nngraph'
require 'Reparametrize'
require 'LinearVA'
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

gen_hid_dim = 8
hid_dim = 1000
out_dim = 1
--Discrim
local input = nn.Identity()()
local hid_lin = nn.Linear(gen_hid_dim,hid_dim)
local hid = nn.ReLU()(hid_lin(input))
local out_lin = nn.Linear(hid_dim,out_dim)
local output = nn.Sigmoid()(out_lin(hid))
network = nn.gModule({input},{output})
--Gen
local input = nn.Identity()()
local hid = nn.ReLU()(nn.LinearVA(in_dim,hid_dim)(input))
local mu = nn.LinearVA(hid_dim,gen_hid_dim)(hid)
local sigma = nn.LinearVA(hid_dim,gen_hid_dim)(hid)
local gauss = nn.Reparametrize(gen_hid_dim){mu,sigma}
local output =nn.ReLU()(nn.LinearVA(gen_hid_dim,in_dim)(gauss))
gen_network = nn.gModule({input},{output,gauss})

--full
local input = nn.Identity()()
local recon,hid = gen_network(input):split(2)
local disc = network(nn.GradientReversal()(hid))
full_network = nn.gModule({input},{recon,disc})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local mse_crit = nn.MSECriterion()
local net_reward = 0
local mb_dim = 32
local data = torch.zeros(mb_dim,gen_hid_dim)
local target = torch.zeros(mb_dim,1)
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)
--
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:training()
    
    local samples = torch.randperm(not8:size(1))
    local gen_data = torch.zeros(mb_dim/2,in_dim)
    for i=1,mb_dim/2 do
        gen_data[i] = not8[samples[i] ]
    end
    target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2)
    target[{{mb_dim/2+1,-1}}] = torch.zeros(mb_dim/2)
    data[{{1,mb_dim/2}}] = torch.randn(mb_dim/2,gen_hid_dim)
    data[{{mb_dim/2+1,-1}}]  = gen_network:forward(gen_data)[2]

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

    local gen_data = torch.zeros(mb_dim,in_dim)
    local samples = torch.randperm(not8:size(1))
    for i=1,mb_dim do
        gen_data[i] = not8[samples[i] ]
    end
    target:zero()
    local recon,disc = unpack(full_network:forward(gen_data))
    local recon_loss = mse_crit:forward(recon,gen_data)
    local recon_grad = mse_crit:backward(recon,gen_data)
    local disc_loss = bce_crit:forward(disc,target)
    local disc_grad = bce_crit:backward(disc,target)
    full_network:backward(gen_data,{recon_grad,disc_grad})
    return loss,dw
end
config = {
    learningRate  = 1e-3
    }
local num_steps = 1e6
local refresh = 1e3
local cumloss =0 
local plot1 = gnuplot.figure()
local plot2 = gnuplot.figure()
for i=1,num_steps do
    for k=1,5 do
        x,batchloss = optim.rmsprop(train_dis,w,config)
    end
    x,batchloss = optim.rmsprop(train_gen,w,config)
    cumloss = cumloss + batchloss[1]
    if i %refresh == 0 then
        print(i,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        gnuplot.figure(plot1)
        samples1 = torch.randperm(not8:size(1))
        samples2 = torch.randperm(notnot8:size(1))
        local data = torch.zeros(mb_dim,in_dim)
        for i=1,mb_dim do
            if i<=mb_dim/2 then
                data[i] = not8[samples1[i] ]
            else
                data[i] = notnot8[samples2[i] ]
            end 
        end
        pic,output = unpack(full_network:forward(data))
        output = output:clone()
        gnuplot.imagesc(pic[1]:reshape(28,28))
        output2 = network:forward(torch.randn(mb_dim/2,gen_hid_dim))
        print(output2:mean(),output[{{1,mb_dim/2}}]:mean(),output[{{mb_dim/2+1,-1}}]:mean())
        net_reward = 0
        cumloss = 0
    end
end
--]]

