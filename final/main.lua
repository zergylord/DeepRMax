require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
--require 'hdf5'
require 'cunn'
require 'util.BCE'
--torch.manualSeed(123)
--cutorch.manualSeed(123)
torch.setnumthreads(1)
local timer = torch.Timer()


--set hyper parameters-------------
--use_rmax = true
rmax = .1
use_qnet = true
q_config = {
    learningRate  = 1e-3
    }
--use_egreedy = true
epsilon = .1
use_target_network = true
target_refresh = 1e3
--use_mnist = true
alpha = .1
gamma = .9
net_reward = 0
refresh = 5e2
num_steps = 1e5


--select environment---------------------------------------------
--require 'environments.combolock'
--require 'environments.grid'
require 'environments.atari'
env.setup{refresh=refresh,num_steps=num_steps}

--select exploration method------------------------------------
require 'models.PPE'
--require 'train_pred_GAN.lua'
setup(env)


--setup value function------------------------------------
if use_qnet then
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
else
    Q = torch.zeros(num_state,env.act_dim)
end






--setup experience replay------------------------------
D = {}
D.size = 1e5
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.r = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.term = torch.zeros(D.size)
if env.byte_storage then
    D.obs = torch.ByteTensor(D.size,in_dim)--s digit
    D.obsPrime = torch.ByteTensor(D.size,in_dim)--sPrime digit
else
    D.obs = torch.zeros(D.size,in_dim)--s digit
    D.obsPrime = torch.zeros(D.size,in_dim)--sPrime digit
end
D.i = 1
local get_data = function(data,action_data,dataPrime)
    num = data:size(1)
    for i=1,num do
        if env.process_for_retrieval then
            data[i] = env.process_for_retrieval(D.obs[mb_ind[i]])
            if env.get_pred_state then
                dataPrime[i] = env.get_pred_state(env.process_for_retrieval(D.obsPrime[mb_ind[i]]))
            else
                dataPrime[i] = env.process_for_retrieval(D.obsPrime[mb_ind[i]])
            end
        else
            data[i] = D.obs[mb_ind[i]]
            dataPrime[i] = D.obsPrime[mb_ind[i]]
        end
        action_data[i] = env.get_action(D.a[mb_ind[i] ])
        
    end
end
set_data_func(get_data)
final_time = -1
state = torch.zeros(mb_dim*env.act_dim,in_dim):cuda()
statePrime = torch.zeros(mb_dim,in_dim):cuda()
action = torch.zeros(mb_dim*env.act_dim,env.act_dim):cuda()
target = torch.zeros(mb_dim*env.act_dim):cuda()
target_mask = torch.zeros(mb_dim*env.act_dim,1):byte()
act_grad = torch.zeros(env.act_dim*mb_dim,env.act_dim):cuda()
aind = torch.LongTensor(mb_dim*env.act_dim,1):cuda()
for a=1,env.act_dim do
    for i=1,mb_dim do
        aind[mb_dim*(a-1)+i] = a
    end
end
--main loop------------------------------------------------------
cumloss =0 
s_obs,s = env.reset()
for t=1,num_steps do
    r = 0
    --select action
    if use_qnet then
        local vals = q_network:forward(s_obs:cuda())
        _,a = vals:max(2)
        a = a[1]
        if use_egreedy then
            if torch.rand(1)[1] < .1 then
                a[1] = torch.random(env.act_dim)
            end
        end
    else
        _,a = torch.max(Q[s],1)
        if use_egreedy then
            if torch.rand(1)[1] < .1 then
                a[1] = torch.random(env.act_dim)
            end
        end
    end
    a = a[1]

    --perform action
    r,sPrime_obs,sPrime,term = env.step(a)
    net_reward = net_reward + r

    --TODO:roll back into env
    env.update_step_stats(s,a)

    --record history
    D.s[D.i] = s
    D.a[D.i] = a
    D.r[D.i] = r
    D.sPrime[D.i] = sPrime
    if term then
        D.term[D.i] = 1
    else
        D.term[D.i] = 0
    end
    if env.process_for_storage then
        D.obs[D.i] = env.process_for_storage(s_obs)
        D.obsPrime[D.i] = env.process_for_storage(sPrime_obs)
    else
        D.obs[D.i] = s_obs:clone() 
        D.obsPrime[D.i] = sPrime_obs:clone() 
    end
    D.i = (D.i % D.size) + 1

    --update model params
    if t > mb_dim then
        --gotta re-perm if runnning train_dis multiple times
        mb_ind = torch.randperm(math.min(t,D.size))
        local mask = torch.zeros(D.size,1):byte()
        action:zero()
        for i =1,mb_dim do
            mask[mb_ind[i] ] = 1
            for a = 1,env.act_dim  do
                action[mb_dim*(a-1)+i][a] = 1
            end
        end
        mind = mask:nonzero()
        if use_rmax then
            --update adver nets
            _,batchloss = optim.adam(train,w,config)
            cumloss = cumloss + batchloss[1]
        end
        --update Q
        local x,y 
        local s,a,r,sPrime,a_actual
        s = D.s[mask:squeeze()]
        s = s:repeatTensor(env.act_dim)
        sPrime = D.sPrime[mask:squeeze()]
        r = D.r[mask:squeeze()]
        term = D.term[mask:squeeze()]
        a_actual = D.a[mask:squeeze()]
        for i=1,mb_dim do
            if env.process_for_retrieval then
                local cur_state = env.process_for_retrieval(D.obs[mind[i][1] ])
                for a=1,env.act_dim do
                    state[mb_dim*(a-1)+i]:copy(cur_state)
                end
                statePrime[i] = env.process_for_retrieval(D.obsPrime[mind[i][1] ])
            else
                for a=1,env.act_dim do
                    state[mb_dim*(a-1)+i] = D.obs[mind[i][1] ]
                end
                statePrime[i] = D.obsPrime[mind[i][1] ]
            end
        end
        if use_qnet then
            if use_target_network then
                --_,qind = q_network:forward(statePrime):max(2)
                --qPrime = target_network:forward(statePrime):gather(2,qind)
                qPrime,qind = target_network:forward(statePrime):max(2)
            else
                qPrime,qind = q_network:forward(statePrime):max(2)
            end
        else
            Q_clone = Q:clone()
        end

        local possible = {state,action}

        if use_rmax then
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
                    env.update_replay_stats(s[i],a,chance_unknown)

                    if  unknown then
                        if a == a_actual[i] then
                            known_flag = false
                        end
                        target[ind] = rmax
                        target_mask[ind] = 1
                    end
                end
            end
            --only experienced actions can be updated over threshold
            if known_flag then
                --print('known!')
                local ind = mb_dim*(a_actual[i]-1)+i
                target_mask[ind] = 1
                if term[i] == 1 then
                    target[ind] = r[i]
                else
                    if use_qnet then
                        target[ind] = r[i]+gamma*torch.max(qPrime[i])
                    else
                        target[ind] = r[i]+gamma*torch.max(Q_clone[sPrime[i]])
                    end
                end
            end
        end
        --update value function----------------------------
        if use_qnet then
            local function q_train(x)
            if x ~= q_w then
                q_w:copy(x)
            end
            q_network:zeroGradParameters()
            local o = q_network:forward(state)
            local used = o:gather(2,aind)
            local loss = mse_crit:forward(used,target)
            local grad = mse_crit:backward(used,target)
            act_grad:zero()
            for i=1,mb_dim*env.act_dim do
                if target_mask[i][1] == 1 then
                    act_grad[i][aind[i][1]] = grad[i]
                end
            end
            q_network:backward(state,act_grad)
            return loss,q_dw
            end
            _,batchloss = optim.adam(q_train,q_w,q_config)
            if use_target_network and t % target_refresh == 0 then
                target_network = q_network:clone()
            end

        else
            for i=1,mb_dim do
                for a=1,env.act_dim do
                    local ind = mb_dim*(a-1)+i
                    if target_mask[ind][1] == 1 then
                        Q[s[i] ][a] = (1-alpha)*Q[s[i] ][a] + (alpha)*target[ind]
                    end
                end
            end
        end
    end
    s = sPrime
    s_obs = sPrime_obs:clone() 
    if t % refresh == 0 then
        env.get_info(t,network,err_network,pred_network,q_network)


        torch.save('w.t7',w)

        print(t,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        timer:reset()
        cumloss = 0
        if num_state and net_reward/refresh > ((1/num_state)*(.8)) then
            final_time = t
            break
        end
        net_reward = 0
        collectgarbage()
    end

end




