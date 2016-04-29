require 'nngraph'
require 'cunn'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'BCE'
--conditions:
--1 -> stochastic cutoff
--2 -> deterministic cutoff
--3 -> predict making cutoff, not err directly
condition = 1
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
    in_dim = num_state or 30
end
thresh = .3 --1.1 --in_dim/20
act_dim = 4
fact_dim = 1000
hid_dim = 1000
mb_dim = 320 --320
config = {learningRate = 1e-3}
dropout = .5

--setup pred_network
input = nn.Identity()()
action = nn.Identity()()
--factor = (nn.ReLU()(nn.Linear(fact_dim,hid_dim)(nn.CMulTable(){nn.Linear(in_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)})))
factor = (((nn.CMulTable(){nn.Linear(in_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)})))
output = nn.Sigmoid()(nn.Linear(hid_dim,in_dim)(factor))
pred_network = nn.gModule({input,action},{output})
pred_network = pred_network:cuda()
--error pred network
input = nn.Identity()()
action = nn.Identity()()
pred = nn.Identity()()
hid = nn.Dropout(dropout)(nn.ReLU()(nn.CAddTable(){nn.Linear(in_dim,hid_dim,false)(input),nn.Linear(act_dim,hid_dim,false)(action),nn.Linear(in_dim,hid_dim)(pred)}))
--[[ fancy doubly factored net
factorSA = nn.CMulTable(){nn.Linear(in_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)}
int_hid1 = nn.ReLU()(nn.Linear(fact_dim,hid_dim)(factorSA))
factor = nn.CMulTable(){nn.Linear(hid_dim,fact_dim)(int_hid1),nn.Linear(in_dim,fact_dim)(pred)}
hid = nn.ReLU()(nn.Linear(fact_dim,hid_dim)(factor))

--]]
if condition == 3 then
    output = nn.Sigmoid()(nn.Linear(hid_dim,1,false)(hid))
else
    output = (nn.Linear(hid_dim,1,false)(hid))
end
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
bce_crit = nn.BCECriterion():cuda()

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
    local loss = bce_crit:forward(o,dataPrime)
    local grad = bce_crit:backward(o,dataPrime)
    pred_network:backward({data,action_data},grad)
    --t_err = torch.pow(o-dataPrime,2):sum(2)
    t_err = BCE(o,dataPrime)
    local o_err = err_network:forward{data,action_data,o}
    local err_grad
    if condition == 3 then
        t_err = t_err:gt(thresh)
        loss = loss + bce_crit:forward(o_err,t_err)
        err_grad =  bce_crit:backward(o_err,t_err)
    else
        loss = loss + mse_crit:forward(o_err,t_err)
        err_grad =  mse_crit:backward(o_err,t_err)
    end
    err_network:backward({data,action_data,o},err_grad)
    return loss,dw
end
--return yes/no and value for storage
if condition == 1 then
    get_knownness = function(output,ind)
        if ind then
            local chance_unknown = output[ind][1]
            return chance_unknown > torch.rand(1)[1]*(thresh/10), math.min(1,math.max(0,chance_unknown/(thresh/10)))
        else
            return output/(thresh/10)
        end
        
    end
elseif condition == 2 then
    get_knownness = function(output,ind)
        if ind then
            local chance_unknown = output[ind][1]
            return chance_unknown > thresh, math.min(1,math.max(0,chance_unknown/thresh))
        else
            return output - thresh
        end
    end
elseif condition == 3 then
    get_knownness = function(output,ind)
        if ind then
            local chance_unknown = output[ind][1]
            return chance_unknown > torch.rand(1)[1], chance_unknown
        else
            return output
        end
    end
end
standard = function()
    num_state = 30
    --setup MDP
    S = torch.eye(num_state)
    --S = torch.tril(torch.ones(num_state,num_state))
    A = torch.eye(act_dim)
    T = torch.ones(num_state,act_dim)
    correct = torch.ones(num_state)
    for i = 1,num_state-1 do
        action = torch.random(act_dim)
        T[i][action] = i+1
        correct[i] = action
    end
    config = {
        learningRate  = 1e-3
        }
    all_statePrime = torch.zeros(num_state*act_dim,in_dim)
    all_state = torch.zeros(num_state*act_dim,in_dim)
    all_action = torch.zeros(num_state*act_dim,act_dim)
    for s=1,num_state do
        for a=1,act_dim do
            all_state[act_dim*(s-1)+a] = S[s]
            all_action[act_dim*(s-1)+a] = A[a]
            all_statePrime[act_dim*(s-1)+a] = S[T[s][a]]
        end
    end
    weighting = torch.linspace(1,num_state,num_state):pow(-2)
    pred_err = torch.ones(act_dim*num_state)
    local get_data = function(data,action_data,dataPrime)
        local num = data:size(1)
        --
        local sample,s,a
        sample = torch.multinomial(weighting,num,true)
        --]]
        for i =1,num do
            --[[
            local sample,s,a
            local bad_sample = true
            while bad_sample do
                sample = torch.multinomial(weighting,1)
                s = sample[1]
                a = torch.random(act_dim)
                bad_sample = pred_err[act_dim*(s-1)+a] < .5
            end
            --]]
            --
            s = sample[i]
            a = torch.random(act_dim)
            --]]

            data[i] = S[s]
            action_data[i] = A[a]
            dataPrime[i] = S[T[s][a] ]
        end
    end
    set_data_func(get_data)
    for i = 1,1e6 do
        optim.adam(train,w,config)
        if i % 1e3==0 then
            network:forward{all_state:cuda(),all_action:cuda()}
            pred_err[{{}}] = network.output:double()
            gnuplot.imagesc(all_statePrime-pred_network.output:double())
            local worst =torch.max(all_statePrime-pred_network.output:double())
            print(i,worst)
            if worst < .5 then
                return
            end
        end
    end
end
--standard()
