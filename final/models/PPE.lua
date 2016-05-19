require 'nngraph'
require 'cunn'
require 'optim'
require 'distributions'
require 'gnuplot'
require '../util/BCE'
--conditions:
--1 -> stochastic cutoff
--2 -> deterministic cutoff
--3 -> predict making cutoff, not err directly
condition = 1
setup = function(env)
in_dim = D.state_dim*D.num_frames
state_dim = env.state_dim
thresh = .3 --1.1 --in_dim/20
act_dim = env.act_dim
fact_dim = 2000
hid_dim = 1000 --1000
config = {learningRate = 1e-3}
dropout = .5

if env.spatial then
    --setup pred_network
    input = nn.Identity()()
    view = nn.View(-1,D.num_frames,84,84)(input)
    action = nn.Identity()()
    conv1 = nn.ReLU()(nn.SpatialConvolutionMM(D.num_frames+1,64,6,6,2,2)(view))
    conv2 = nn.ReLU()(nn.SpatialConvolutionMM(64,64,6,6,2,2,2,2)(conv1))
    conv3 = nn.ReLU()(nn.SpatialConvolutionMM(64,64,6,6,2,2,2,2)(conv2))
    num_conv = 64*10*10
    enc = nn.ReLU()(nn.Linear(num_conv,hid_dim)(nn.View(num_conv)(conv3)))
    dec = nn.Linear(fact_dim,num_conv)(nn.CMulTable(){nn.Linear(hid_dim,fact_dim,false)(enc),nn.Linear(act_dim,fact_dim,false)(action)})
    deconv3 = nn.ReLU()(nn.SpatialFullConvolution(64,64,2,2,2,2)(nn.View(64,10,10)(dec)))
    deconv2 = nn.ReLU()(nn.SpatialFullConvolution(64,64,2,2,2,2)(deconv3))
    deconv1 = nn.SpatialFullConvolution(64,1,6,6,2,2)(deconv2)
    out = nn.Sigmoid()(deconv1)
    --out = deconv1
    pred_network = nn.gModule({input,action},{out})
    pred_network = pred_network:cuda()
    --error pred network
    input = nn.Identity()()
    in_view = nn.View(-1,D.num_frames+1,84,84)(input)
    action = nn.Identity()()
    pred = nn.Identity()()
    pred_view = nn.View(-1,1,84,84)(pred)
    conv1 = nn.ReLU()(nn.SpatialConvolutionMM(D.num_frames+2,64,6,6,2,2)(nn.JoinTable(2){in_view,pred_view}))
    conv2 = nn.ReLU()(nn.SpatialConvolutionMM(64,64,6,6,2,2,2,2)(conv1))
    s_conv = nn.View(num_conv)(nn.ReLU()(nn.SpatialConvolutionMM(64,64,6,6,2,2,2,2)(conv2)))
    hid = nn.Dropout(dropout)(nn.ReLU()(nn.CAddTable(){nn.Linear(num_conv,hid_dim,false)(s_conv),nn.Linear(act_dim,hid_dim,false)(action)}))
    --[[
    conv1 = nn.ReLU()(nn.SpatialConvolutionMM(1,64,6,6,2,2)(pred))
    conv2 = nn.ReLU()(nn.SpatialConvolutionMM(64,64,6,6,2,2,2,2)(conv1))
    sPrime_conv = nn.View(num_conv)(nn.ReLU()(nn.SpatialConvolutionMM(64,64,6,6,2,2,2,2)(conv2)))
    hid = nn.Dropout(dropout)(nn.ReLU()(nn.CAddTable(){nn.Linear(num_conv,hid_dim,false)(s_conv),nn.Linear(act_dim,hid_dim,false)(action),nn.Linear(num_conv,hid_dim)(sPrime_conv)}))
    --]]
    if condition == 3 then
        output = nn.Sigmoid()(nn.Linear(hid_dim,1,false)(hid))
    else
        output = (nn.Linear(hid_dim,1,false)(hid))
    end
    err_network = nn.gModule({input,action,pred},{output})
    err_network = err_network:cuda()
else
    --setup pred_network
    input = nn.Identity()()
    action = nn.Identity()()
    --factor = (nn.ReLU()(nn.Linear(fact_dim,hid_dim)(nn.CMulTable(){nn.Linear(in_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)})))
    factor = (((nn.CMulTable(){nn.Linear(in_dim,hid_dim)(input),nn.Linear(act_dim,hid_dim)(action)})))
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
end

--full network (mainly for gathering params)
input = nn.Identity()()
action = nn.Identity()()
output = err_network{input,action,pred_network{input,action}}
network = nn.gModule({input,action},{output})
network = network:cuda()

w,dw = network:getParameters()
mse_crit = nn.MSECriterion():cuda()
bce_crit = nn.BCECriterion():cuda()
data = torch.zeros(opt.mb_dim,in_dim):cuda()
dataPrime = torch.zeros(opt.mb_dim,state_dim):cuda()
action_data = torch.zeros(opt.mb_dim,act_dim):cuda()
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
    network:zeroGradParameters()
    data_func(data,action_data,dataPrime)
    local o = pred_network:forward{data,action_data}
    local loss = bce_crit:forward(o,dataPrime)
    local grad = bce_crit:backward(o,dataPrime)
    pred_network:backward({data,action_data},grad)
    --t_err = torch.pow(o-dataPrime,2):sum(2)
    if o:dim() >2 then
        t_err = BCE(o:view(mb_dim,state_dim),dataPrime:view(mb_dim,state_dim))
    else
        t_err = BCE(o,dataPrime)
    end

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
            pred_err[{{}}] = network.output:float()
            gnuplot.imagesc(all_statePrime-pred_network.output:float())
            local worst =torch.max(all_statePrime-pred_network.output:float())
            print(i,worst)
            if worst < .5 then
                return
            end
        end
    end
end
atari = function()
    require 'image'
    num_state = 84*84
    use_atari = true
    num_hist = 3
    setup()
    --setup MDP
    raw,actions = unpack(torch.load('frames.dat'))
    raw = raw:float()/256
    --raw = (raw:float()-raw:float():mean())/128
    frames = torch.zeros(raw:size(1),num_hist+1,84,84)
    for i=num_hist+1,raw:size(1) do
        for j=1,num_hist+1 do
            frames[i][j] = image.scale(raw[i-j+1][{{31,190}}],84,84)
        end
    end
    config = {
        learningRate  = 1e-6
        }
    pred_err = torch.ones(act_dim*num_state)
    weighting = torch.ones(frames:size(1)-1)
    A = torch.eye(act_dim)
    local get_data = function(data,action_data,dataPrime)
        local num = data:size(1)
        local sample,s,a
        sample = torch.multinomial(weighting,num,true)
        for i =1,num do
            data[i] = frames[sample[i] ]
            action_data[i] = A[actions[sample[i]]]
            dataPrime[i] = frames[sample[i]+1][1]
        end
    end
    set_data_func(get_data)
    cumloss = 0
    for i = 1,1e7 do
        _,batchloss = optim.adam(train,w,config)
        cumloss = cumloss + batchloss[1]
        if i % 1e4==0 then
            gnuplot.figure(1)
            gnuplot.imagesc(pred_network.output[1]:view(84,84))
            gnuplot.figure(2)
            gnuplot.imagesc(dataPrime[1]:view(84,84))
            print(i,cumloss,w:norm(),dw:norm())
            cumloss = 0
        end
    end
end
--atari()
