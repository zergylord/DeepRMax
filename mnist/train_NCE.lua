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
    in_dim = num_state
end
hid_dim = 1000
gen_hid_dim = 1000
dropout = 0 --0.5
mb_dim = 320
out_dim = 1
fact_dim = 10
--Discrim
local input = nn.Identity():cuda()()
local action = nn.Identity():cuda()()
hid_lin = nn.Linear(in_dim,hid_dim):cuda()
hid = nn.Dropout(dropout):cuda()(nn.ReLU():cuda()(hid_lin(input)))
local factor = nn.Dropout(dropout):cuda()(nn.CMulTable():cuda(){nn.Linear(hid_dim,fact_dim):cuda()(hid),nn.Linear(act_dim,fact_dim):cuda()(action)})
local prob = nn.SoftPlus():cuda()(nn.Linear(fact_dim,out_dim):cuda())
local denom = nn.AddConstant(4):cuda()(prob)
local output = nn.CDivTable():cuda(){prob,denom}
network = nn.gModule({input,action},{output,prob})


w,dw = network:getParameters()
print(w:type())
local timer = torch.Timer()
bce_crit = nn.BCECriterion():cuda()
local net_reward = 0
data = torch.zeros(mb_dim,in_dim):cuda()
action_data = torch.zeros(mb_dim,act_dim):cuda()
dis_target = torch.zeros(mb_dim,1):cuda()
dis_target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2):cuda()

--[[
--Public
--single training step
--assumes get_data has been set to a minibatch generating function
--]]

train = function(x)
    if x ~= w then
        w:copy(x)
    end
    network:zeroGradParameters()
    network:training()
    data_func(data[{{1,mb_dim/2}}],action_data[{{1,mb_dim/2}}])

    local output
    action_data[{{mb_dim/2+1,-1}}] = distributions.cat.rnd(mb_dim/2,torch.ones(act_dim),{categories=torch.eye(act_dim)})

    output = network:forward{data,action_data}
    last_compare = output

    local loss = bce_crit:forward(output,dis_target)
    local grad = bce_crit:backward(output,dis_target)
    network:backward({data,action_data},{grad,torch.zeros(mb_dim,1)})
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
config = {
    learningRate  = 1e-3
    }
