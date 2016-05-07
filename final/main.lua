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
require 'environments.grid'
env.setup()

--select exploration method------------------------------------
require 'models.PPE'
--require 'train_pred_GAN.lua'
setup()


--setup value function------------------------------------
if use_qnet then
    local hid_dim = 100
    local input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
    local output =nn.Linear(hid_dim,act_dim)(hid)
    q_network = nn.gModule({input},{output})
    q_network = q_network:cuda()
    q_w,q_dw = q_network:getParameters()
    if use_target_network then
        --target network
        local input = nn.Identity()()
        local hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
        local output =nn.Linear(hid_dim,act_dim)(hid)
        target_network = nn.gModule({input},{output})
        target_network = target_network:cuda()
        target_w,target_dw = target_network:getParameters()
        target_w:copy(q_w)

    end
    mse_crit = nn.MSECriterion():cuda()
else
    Q = torch.zeros(num_state,act_dim)
end





--init viz variables----------------------------------
bonus_hist = torch.zeros(num_steps/refresh)
C = torch.zeros(mb_dim)
neg_entropy = torch.zeros(num_state,act_dim)
hist_total = torch.zeros(num_state,act_dim)
visits = torch.zeros(num_state)
visits_over_time = torch.zeros(num_steps/refresh,num_state)
neg_entropy_over_time = torch.zeros(num_steps/refresh,num_state*act_dim)
neg_entropy_over_time_norm = torch.zeros(num_steps/refresh,num_state*act_dim)
neg_entropy_over_time_per_state = torch.zeros(num_steps/refresh,num_state)
neg_entropy_over_time_per_state_norm = torch.zeros(num_steps/refresh,num_state)
sa_visits = torch.zeros(num_state,act_dim)

--setup experience replay------------------------------
D = {}
D.size = 1e5
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.r = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.obs = torch.zeros(D.size,in_dim)--s digit
D.obsPrime = torch.zeros(D.size,in_dim)--sPrime digit
D.i = 1
local get_data = function(data,action_data,dataPrime)
    num = data:size(1)
    for i=1,num do
        data[i] = D.obs[mb_ind[i]]
        if dataPrime then
            dataPrime[i] = D.obsPrime[mb_ind[i]]
        end
        action_data[i] = env.get_action(D.a[mb_ind[i] ])
        
    end
end
set_data_func(get_data)
final_time = -1
state = torch.zeros(mb_dim*act_dim,in_dim)
statePrime = torch.zeros(mb_dim,in_dim)
aind = torch.LongTensor(mb_dim*act_dim,1)
for a=1,act_dim do
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
        local vals = q_network:forward(s_obs:view(1,in_dim):cuda())
        _,a = vals:max(2)
        a = a[1]
        if use_egreedy then
            if torch.rand(1)[1] < .1 then
                a[1] = torch.random(act_dim)
            end
        end
    else
        _,a = torch.max(Q[s],1)
        if use_egreedy then
            if torch.rand(1)[1] < .1 then
                a[1] = torch.random(act_dim)
            end
        end
    end
    a = a[1]

    --perform action
    r,sPrime_obs,sPrime = env.step(s,a)
    net_reward = net_reward + r

    sa_visits[s][a] = sa_visits[s][a] + 1
    visits[sPrime] = visits[sPrime] + 1

    --record history
    D.s[D.i] = s
    D.a[D.i] = a
    D.r[D.i] = r
    D.sPrime[D.i] = sPrime
    D.obs[D.i] = s_obs:clone() 
    D.obsPrime[D.i] = sPrime_obs:clone() 
    D.i = (D.i % D.size) + 1

    --update model params
    if t > mb_dim then
        --gotta re-perm if runnning train_dis multiple times
        mb_ind = torch.randperm(math.min(t,D.size))
        local mask = torch.zeros(D.size,1):byte()
        local action = torch.zeros(mb_dim*act_dim,act_dim)
        for i =1,mb_dim do
            mask[mb_ind[i] ] = 1
            for a = 1,act_dim  do
                action[mb_dim*(a-1)+i][a] = 1
            end
        end
        mind = mask:nonzero()
        --update adver nets
        _,batchloss = optim.adam(train,w,config)
        cumloss = cumloss + batchloss[1]
        --update Q
        local x,y,target 
        local s,a,r,sPrime,a_actual
        s = D.s[mask:squeeze()]
        s = s:repeatTensor(act_dim)
        sPrime = D.sPrime[mask:squeeze()]
        r = D.r[mask:squeeze()]
        a_actual = D.a[mask:squeeze()]
        for i=1,mb_dim do
            for a=1,act_dim do
                state[mb_dim*(a-1)+i] = D.obs[mind[i][1] ]
            end
            statePrime[i] = D.obsPrime[mind[i][1] ]
        end
        if use_qnet then
            if use_target_network then
                --_,qind = q_network:forward(statePrime:cuda()):max(2)
                --qPrime = target_network:forward(statePrime:cuda()):gather(2,qind)
                qPrime,qind = target_network:forward(statePrime:cuda()):max(2)
            else
                qPrime,qind = q_network:forward(statePrime:cuda()):max(2)
            end
        else
            Q_clone = Q:clone()
        end

        local possible = {state:cuda(),action:cuda()}

        C = network:forward(possible)
    
        
        target = torch.zeros(mb_dim*act_dim)
        target_mask = torch.zeros(mb_dim*act_dim,1):byte()
        for i=1,mb_dim do
            --you can experience all actions under threshold, since they all go to heaven!
            local known_flag = true
            for a = 1,act_dim do
                local ind = mb_dim*(a-1)+i
                hist_total[s[i] ][a] = hist_total[s[i] ][a] + 1
                local unknown, chance_unknown = get_knownness(C,ind)
                neg_entropy[s[i] ][a] = neg_entropy[s[i] ][a] + chance_unknown

                if  unknown then
                    if a == a_actual[i] then
                        known_flag = false
                    end
                    local r = rmax
                    target[ind] = r
                    target_mask[ind] = 1
                end
            end
            --only experienced actions can be updated over threshold
            if known_flag then
                --print('known!')
                local ind = mb_dim*(a_actual[i]-1)+i
                hist_total[s[i] ][a_actual[i]] = hist_total[s[i] ][a_actual[i]] + 1
                target_mask[ind] = 1
                if s[i] == num_state then
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
            local o = q_network:forward(state:cuda())
            local used = o:gather(2,aind:cuda())
            local loss = mse_crit:forward(used,target:cuda())
            local grad = mse_crit:backward(used,target:cuda())
            local act_grad = torch.zeros(act_dim*mb_dim,act_dim):cuda()
            for i=1,mb_dim*act_dim do
                if target_mask[i][1] == 1 then
                    act_grad[i][aind[i][1]] = grad[i]
                end
            end
            q_network:backward(state,act_grad)
            return loss,q_dw
            end
            _,batchloss = optim.adam(q_train,q_w,q_config)
            if use_target_network and t % target_refresh == 0 then
                target_w:copy(q_w)
            end

        else
            for i=1,mb_dim do
                for a=1,act_dim do
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
        gnuplot.figure(1)
        gnuplot.raw("set multiplot layout 2,5 columnsfirst")


        if use_qnet then
            local Q
            if use_mnist then
                local data = torch.zeros(num_state,in_dim):cuda()
                for s = 1,num_state do
                    data[s] = digit[s][torch.random(digit[s]:size(1))]
                end
                Q = q_network:forward(data)
            else
                Q = q_network:forward(env.get_all_states():cuda())
            end

            gnuplot.raw("set title 'Q-values' ")
            gnuplot.raw('set xrange [' .. .5 .. ':' .. num_state+.5 .. '] noreverse')
            gnuplot.raw('set yrange [*:*] noreverse')
            gnuplot.plot({Q:mean(2)},{Q:max(2):double()},{Q:min(2):double()})
        else
            gnuplot.raw("set title 'Q-values' ")
            gnuplot.raw('set xrange [' .. .5 .. ':' .. num_state+.5 .. '] noreverse')
            gnuplot.raw('set yrange [0:1] noreverse')
            gnuplot.plot({Q:mean(2)},{Q:max(2):double()},{Q:min(2):double()})
        end


        
        gnuplot.raw("set title '% R-Max bonus over time' ")
        gnuplot.raw('set xrange [' .. 0 .. ':' .. t/refresh+1 .. '] noreverse')
        gnuplot.raw('set yrange [0:.1] noreverse')
        local bonus = neg_entropy:cdiv(hist_total+1)
        print(bonus:mean())
        bonus_hist[t/refresh] = bonus:mean()
        gnuplot.plot(bonus_hist[{{1,t/refresh}}])
        
        
        
        neg_entropy_over_time[t/refresh] = bonus
        neg_entropy_over_time_per_state[t/refresh] = bonus:mean(2)
        neg_entropy_over_time_norm[t/refresh] = bonus:div(bonus:max())
        neg_entropy_over_time_per_state_norm[t/refresh] = bonus:mean(2):div(bonus:mean(2):max())
        gnuplot.raw("set title 'bonus % over time' ")
        gnuplot.imagesc(neg_entropy_over_time[{{1,t/refresh}}])
        gnuplot.raw("set title 'bonus % over time per state' ")
        gnuplot.imagesc(neg_entropy_over_time_per_state[{{1,t/refresh}}])
        gnuplot.raw("set title 'bonus % over time normed' ")
        gnuplot.imagesc(neg_entropy_over_time_norm[{{1,t/refresh}}])
        gnuplot.raw("set title 'bonus % over time per state normed' ")
        gnuplot.imagesc(neg_entropy_over_time_per_state_norm[{{1,t/refresh}}])

        print(visits)
        visits_over_time[t/refresh] = visits

        cur_known = torch.zeros(num_state,act_dim)
        cur_pred = torch.zeros(num_state*act_dim,in_dim)
        cur_actual_pred = torch.zeros(num_state*act_dim,in_dim)
        cur_err = torch.zeros(num_state,act_dim)
        cur_actual_err = torch.zeros(num_state,act_dim)

        local all_state,all_action,all_statePrime = env.get_complete_dataset()
        network:forward{all_state:cuda(),all_action:cuda()}
        cur_pred = pred_network.output:double()
        cur_actual_pred = all_statePrime
        cur_err = get_knownness(err_network.output):reshape(num_state,act_dim)
        cur_actual_err = BCE(cur_pred,cur_actual_pred):reshape(num_state,act_dim)
        
        gnuplot.raw("set title 'current pred' ")
        gnuplot.imagesc(cur_pred)
        gnuplot.raw("set title 'pred diff' ")
        gnuplot.imagesc(cur_actual_pred - cur_pred)
        gnuplot.raw("set title 'current pred pred err' ")
        gnuplot.imagesc(cur_err)
        gnuplot.raw("set title 'current actual pred err' ")
        gnuplot.imagesc(cur_actual_err)

        torch.save('w.t7',w)
        


        gnuplot.raw('unset multiplot')

        print(t,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        gen_count = 0
        timer:reset()
        cumloss = 0
        if net_reward/refresh > ((1/num_state)*(1-epsilon)) then
            final_time = t
            break
        end
        net_reward = 0
        visits:zero()
        sa_visits:zero()
        hist_total:zero()
        neg_entropy:zero()
        collectgarbage()
    end

end




