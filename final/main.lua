require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'ReplayTable'
--require 'hdf5'
require 'cunn'
require 'util.BCE'
--torch.manualSeed(123)
--cutorch.manualSeed(123)
torch.setnumthreads(1)
local timer = torch.Timer()


--set hyper parameters-------------
update_freq = 4
--use_rmax = true
mb_dim = 32 --320
rmax = .1
q_config = {
    learningRate  = 1e-3
    }
refresh = 1e3
num_steps = 1e7
use_egreedy = true
epsilon = (-torch.linspace(0,.9,1e6)+1):cat(torch.ones(num_steps-1e6):mul(.1))
use_target_network = true
target_refresh = 1e4
--use_mnist = true
clip_delta = 1
gamma = .9
net_reward = 0
reward_hist = torch.zeros(num_steps/refresh)


--select environment---------------------------------------------
--require 'environments.combolock'
--require 'environments.grid'
require 'environments.atari'
env.setup{refresh=refresh,num_steps=num_steps}
D = ReplayTable.init(env.in_dim,env.byte_storage)

--select exploration method------------------------------------
require 'models.PPE'
--require 'train_pred_GAN.lua'
setup(env)


--setup value function------------------------------------
local input,output
if env.spatial then
    input = nn.Identity()()
    local view = nn.View(-1,env.num_hist+1,env.image_size[1],env.image_size[2])(input)
    local conv1 = nn.ReLU()(nn.SpatialConvolution(env.num_hist+1,32,8,8,4,4)(view))
    local conv2 = nn.ReLU()(nn.SpatialConvolution(32,64,4,4,2,2)(conv1))
    local conv3 = nn.ReLU()(nn.SpatialConvolution(64,64,3,3,1,1)(conv2))
    local conv_dim = 64*7*7
    local last_hid = nn.ReLU()(nn.Linear(conv_dim,512)(nn.View(-1,conv_dim)(conv3)))
    output = nn.Linear(512,env.act_dim)(last_hid)
else
    local hid_dim = 100
    input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
    local output =nn.Linear(hid_dim,env.act_dim)(hid)
end
q_network = nn.gModule({input},{output})
q_network = q_network:cuda()
q_w,q_dw = q_network:getParameters()
if use_target_network then
    target_network = q_network:clone()
end
mse_crit = nn.MSECriterion():cuda()






--setup experience replay------------------------------
local get_data = function(data,action_data,dataPrime)
    data = mb_s
    action_data = mb_a
    if env.get_pred_state then
        dataPrime = env.get_pred_state(mb_sPrime)
    else
        dataPrime = mb_sPrime
    end
end
set_data_func(get_data)
final_time = -1
possible = {}
possible[1] = torch.zeros(mb_dim*env.act_dim,in_dim):cuda()
possible[2] = torch.zeros(mb_dim*env.act_dim,env.act_dim):cuda()
for i =1,mb_dim do
    for a = 1,env.act_dim  do
        possible[2][mb_dim*(a-1)+i][a] = 1
    end
end
target = torch.zeros(mb_dim,env.act_dim):cuda()
target_mask = torch.zeros(mb_dim,env.act_dim,1):byte()
act_grad = torch.zeros(env.act_dim*mb_dim,env.act_dim):cuda()
aind = torch.LongTensor(mb_dim*env.act_dim,1):cuda()
for a=1,env.act_dim do
    for i=1,mb_dim do
        aind[mb_dim*(a-1)+i] = a
    end
end
local function q_train(x)
    q_network:zeroGradParameters()
    local o = q_network:forward(mb_s)
    --
    local loss = mse_crit:forward(o,target)
    act_grad = mse_crit:backward(o,target)
    --]]
    --[[
    act_grad = -(target-o)
    local loss = 0
    --]]
    act_grad[target_mask:eq(0)] = 0
    if clip_delta then
        act_grad[act_grad:gt(0)] = clip_delta
        act_grad[act_grad:lt(0)] = -clip_delta
    end
    q_network:backward(mb_s,act_grad)
    return loss,q_dw
end
--main loop------------------------------------------------------
cumloss =0 
s = env.reset()
for t=1,num_steps do
    r = 0
    --select action
    local vals = q_network:forward(s:cuda())
    _,a = vals:max(2)
    a = a[1]
    if use_egreedy then
        if torch.rand(1)[1] < epsilon[t] then
            a[1] = torch.random(env.act_dim)
        end
    end
    a = a[1]

    --perform action
    r,sPrime,term = env.step(a)
    if r > 0 then
        r = 1
    elseif r< 0 then
        r = -1
    end
    net_reward = net_reward + r

    --TODO:roll back into env
    env.update_step_stats(s,a)

    --record history
    D:add(s,a,r,sPrime,term)

    --update model params
    if t > mb_dim and t % update_freq == 0 then
        --gotta re-perm if runnning train_dis multiple times
        mb_s,mb_a,mb_r,mb_sPrime,mb_term = D:get_samples(mb_dim)
        if use_rmax then
            --update adver nets
            _,batchloss = optim.adam(train,w,config)
            cumloss = cumloss + batchloss[1]
        end
        --update Q
        if use_target_network then
            --_,qind = q_network:forward(mb_sPrime):max(2)
            --qPrime = target_network:forward(mb_sPrime):gather(2,qind)
            qPrime,qind = target_network:forward(mb_sPrime):max(2)
        else
            qPrime,qind = q_network:forward(mb_sPrime):max(2)
        end

        if use_rmax then
            for i=1,mb_dim do
                for a=1,env.act_dim do
                    possible[1][mb_dim*(a-1)+i] = mb_s[i]
                end
            end
            C = network:forward(possible)
        end
    
        
        target:zero()
        target_mask:zero()
        for i=1,mb_dim do
            --you can experience all actions under threshold, since they all go to heaven!
            local known_flag = true
            if use_rmax then
                for a = 1,env.act_dim do
                    local ind = mb_dim*(a-1)+i
                    local unknown, chance_unknown = get_knownness(C,ind)
                    env.update_replay_stats(mb_s[i],a,chance_unknown)

                    if  unknown then
                        if a == mb_a[i] then
                            known_flag = false
                        end
                        target[i][a] = rmax
                        target_mask[i][a] = 1
                    end
                end
            end
            --only experienced actions can be updated over threshold
            if known_flag then
                --print('known!')
                target_mask[i][mb_a[i] ] = 1
                if mb_term[i] == 1 then
                    target[i][mb_a[i] ] = mb_r[i]
                else
                    target[i][mb_a[i] ] = mb_r[i]+gamma*torch.max(qPrime[i])
                end
            end
        end
        --update value function----------------------------
        _,batchloss = optim.adam(q_train,q_w,q_config)
        if use_target_network and t % target_refresh == 0 then
            target_network = q_network:clone()
        end
    end
    s = sPrime 
    if t % refresh == 0 then
        env.get_info(t,network,err_network,pred_network,q_network)


        torch.save('w.t7',{q_w,w})

        print(t,net_reward/refresh,cumloss,q_w:norm(),q_dw:norm(),timer:time().real)
        timer:reset()
        cumloss = 0
        if num_state and net_reward/refresh > ((1/num_state)*(.8)) then
            final_time = t
            break
        end
        reward_hist[t/refresh] = net_reward
        gnuplot.plot(reward_hist[{{1,t/refresh}}])
        net_reward = 0
        collectgarbage()
    end

end




