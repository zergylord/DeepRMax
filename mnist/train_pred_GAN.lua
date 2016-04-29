require 'nngraph'
require 'cunn'
require 'optim'
require 'distributions'
require 'gnuplot'
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
thresh = 1.1 --in_dim/20
act_dim = 4
fact_dim = 20
pred_hid_dim = 100
err_hid_dim = 100
mb_dim = 320
config = {learningRate = 2e-4, beta1=.5}
dropout = .5

--setup pred_network
input = nn.Identity()()
action = nn.Identity()()
--factor = nn.BatchNormalization(pred_hid_dim)(((nn.CMulTable(){nn.Linear(in_dim,pred_hid_dim)(input),nn.Linear(act_dim,pred_hid_dim)(action)})))
hid1 = nn.BatchNormalization(pred_hid_dim)((nn.ReLU()(nn.CAddTable(){nn.Linear(in_dim,pred_hid_dim,false)(input),nn.Linear(act_dim,pred_hid_dim,false)(action)})))
hid2 = nn.BatchNormalization(pred_hid_dim)(nn.ReLU()(nn.Linear(pred_hid_dim,pred_hid_dim,false)(hid1)))
last_hid = hid2
output = nn.Sigmoid()(nn.Linear(pred_hid_dim,in_dim,false)(last_hid))
pred_network = nn.gModule({input,action},{output})
pred_network = pred_network:cuda()
--error pred network
input = nn.Identity()()
action = nn.Identity()()
pred = nn.Identity()()
--hid = nn.Dropout(dropout)(nn.ReLU()(nn.CAddTable(){nn.Linear(in_dim,err_hid_dim)(input),nn.Linear(act_dim,err_hid_dim)(action),nn.Linear(in_dim,err_hid_dim)(pred)}))
hid = nn.Dropout(dropout)(nn.ReLU()(nn.CAddTable(){nn.Linear(in_dim,err_hid_dim)(input),nn.Linear(act_dim,err_hid_dim,false)(action),nn.Linear(in_dim,err_hid_dim,false)(pred)}))
--hid = nn.LeakyReLU(.2)(nn.CAddTable(){nn.Linear(in_dim,err_hid_dim)(input),nn.Linear(act_dim,err_hid_dim,false)(action),nn.Linear(in_dim,err_hid_dim,false)(pred)})
--[[ fancy doubly factored net
factorSA = nn.CMulTable(){nn.Linear(in_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)}
int_hid1 = nn.ReLU()(nn.Linear(fact_dim,err_hid_dim)(factorSA))
factor = nn.CMulTable(){nn.Linear(err_hid_dim,fact_dim)(int_hid1),nn.Linear(in_dim,fact_dim)(pred)}
hid = nn.ReLU()(nn.Linear(fact_dim,err_hid_dim)(factor))

--]]
output = nn.Sigmoid()(nn.Linear(err_hid_dim,1)(hid))
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
local gen_t = torch.ones(mb_dim,1):cuda()
local dis_t = torch.ones(mb_dim,1):cuda()
dis_t[{{mb_dim/2+1,-1}}] = 0
local flip_t = torch.zeros(mb_dim,1):cuda()
train = function(x)
    network:zeroGradParameters()
    -----------------------train generator-------------------------
    local loss
    --err_network:evaluate()
    data_func(data,action_data,dataPrime)
    local o = network:forward{data,action_data}
    loss = bce_crit:forward(o,gen_t)
    -- backprop through dis
    local err_grad = bce_crit:backward(o,gen_t):clone()
    local _,_,pred_grad = unpack(err_network:backward({data,action_data,pred_network.output},err_grad))
    pred_grad = pred_grad:clone()
    -- flipped grad
    loss = loss + bce_crit:forward(o,flip_t)
    local pre_flip_grad =  bce_crit:backward(o,flip_t):clone()
    local _,_,flip_grad = unpack(err_network:backward({data,action_data,pred_network.output},pre_flip_grad))
    pred_grad = (pred_grad - flip_grad)/2
    --]]
    pred_network:backward({data,action_data},pred_grad)
    --]]
    --[[ supervised term to speed gen learning
    loss = loss + bce_crit:forward(pred_network.output,dataPrime)
    local grad = bce_crit:backward(pred_network.output,dataPrime)
    pred_network:backward({data,action_data},grad)
    --]]
    
    err_network:zeroGradParameters() --dont want to actually change the Discrim weights!
    err_network:training()
    ---------------------train descriminator----------------------
    data_func(data,action_data,dataPrime)
    dataPrime[{{mb_dim/2+1,-1}}] = pred_network:forward{data[{{mb_dim/2+1,-1}}],action_data[{{mb_dim/2+1,-1}}]}
    --dataPrime[{{mb_dim/2+1,-1}}] = pred_network.output[{{mb_dim/2+1,-1}}]:clone()
    local o = err_network:forward{data,action_data,dataPrime}
    loss = loss + bce_crit:forward(o,dis_t)
    local grad = bce_crit:backward(o,dis_t)
    err_network:backward({data,action_data,dataPrime},grad)
    return loss,dw
end
--return yes/no and value for storage
thresh = .6
steep = 100
soft_thresh = function(x) return torch.pow(torch.exp(-(x-thresh)*steep)+1,-1) end
get_knownness = function(output,ind)
    if ind then
        local chance_unknown = soft_thresh(1 - output[ind][1])
        --return chance_unknown > thresh, math.max(0,chance_unknown - thresh)
        return chance_unknown > torch.rand(1)[1], chance_unknown
    else
        return soft_thresh(-output+1)
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
    --weighting = torch.linspace(1,num_state,num_state):pow(-2)
    weighting = torch.ones(num_state)
    --weighting = torch.linspace(1,num_state,num_state):mul(-1.5):exp()
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
        if i % 5e2==0 then
            --w = torch.load('w.t7')
            network:forward{all_state:cuda(),all_action:cuda()}
            local chance_unknown = soft_thresh(-network.output:double()+1)
            pred_err = network.output:double():cat(err_network:forward{all_state:cuda(),all_action:cuda(),all_statePrime:cuda()}:double(),1)

            gnuplot.figure(1)
            gnuplot.imagesc(pred_network.output:double())
            gnuplot.figure(2)
            gnuplot.bar(pred_err)
            gnuplot.axis{'','',0,1}
            local worst =torch.max(all_statePrime-pred_network.output:double())
            print(i,w:norm(),dw:norm())
            if worst < .5 then
                return
            end
        end
    end
end
--standard()

