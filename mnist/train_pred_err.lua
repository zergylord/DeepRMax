require 'nngraph'
require 'cunn'
require 'optim'
require 'distributions'
require 'gnuplot'
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
act_dim = 4
fact_dim = 15
hid_dim = 1000
mb_dim = 320
config = {learningRate = 1e-3}

--setup pred_network
input = nn.Identity()()
action = nn.Identity()()
factor = nn.ReLU()(nn.Linear(fact_dim,hid_dim)(nn.CMulTable(){nn.Linear(in_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)}))
output = nn.Sigmoid()(nn.Linear(hid_dim,in_dim)(factor))
pred_network = nn.gModule({input,action},{output})
pred_network = pred_network:cuda()
--error pred network
input = nn.Identity()()
action = nn.Identity()()
pred = nn.Identity()()
hid = nn.ReLU()(nn.CAddTable(){nn.Linear(in_dim,hid_dim)(input),nn.Linear(act_dim,hid_dim)(action),nn.Linear(in_dim,hid_dim)(pred)})
output = (nn.Linear(hid_dim,1)(hid))
err_network = nn.gModule({input,action,pred},{output})
err_network = err_network:cuda()

--full network (mainly for gathering params)
input = nn.Identity()()
action = nn.Identity()()
output = err_network{input,action,pred_network{input,action}}
network = nn.gModule({input,action},{output})
network = network:cuda()

w,dw = network:getParameters()
mse_crit = nn.MSECriterion():cuda()

data = torch.zeros(mb_dim,in_dim):cuda()
action_data = torch.zeros(mb_dim,act_dim):cuda()
dataPrime = torch.zeros(mb_dim,in_dim):cuda()
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
    network:zeroGradParameters()
    data_func(data,action_data,dataPrime)
    local o = pred_network:forward{data,action_data}
    local loss = mse_crit:forward(o,dataPrime)
    local grad = mse_crit:backward(o,dataPrime)
    pred_network:backward({data,action_data},grad)
    t_err = torch.pow(o-dataPrime,2):sum(2)
    local o_err = err_network:forward{data,action_data,o}
    loss = loss + mse_crit:forward(o_err,t_err)
    local err_grad =  mse_crit:backward(o_err,t_err)
    err_network:backward({data,action_data,o},err_grad)
    return loss,dw
end
--return yes/no and value for storage
get_knownness = function(output,ind)
    local chance_unknown = output[ind][1]
    return chance_unknown > torch.rand(1)[1]*(in_dim/2), math.min(1,math.max(0,chance_unknown/(in_dim/2)))
end
