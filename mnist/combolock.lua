require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
require 'cunn'
require 'BCE'
--torch.manualSeed(123)
--cutorch.manualSeed(123)
torch.setnumthreads(1)
log2 = function(x) return torch.log(x)/torch.log(2) end
--H = function(p) return log2(p):cmul(-p)-log2(-p+1):cmul(-p+1) end
H = function(p) return log2(p)*(-p)-log2(-p+1)*(-p+1) end
noise_mag = 0--.05
temp =  1 --.5

act_dim = 4


s = 1
local timer = torch.Timer()
rmax = .1
--use_qnet = true
--use_egreedy = true
use_target_network = true
--use_mnist = true
A = torch.eye(act_dim)
--A = torch.tril(torch.ones(act_dim,act_dim))
num_state = 30
T = torch.ones(num_state,act_dim)
correct = torch.ones(num_state)
for i = 1,num_state-1 do
    action = torch.random(act_dim)
    T[i][action] = i+1
    correct[i] = action
end
if use_mnist then
    digit = torch.load('digit.t7')
    s_obs = digit[s][torch.random(digit[s]:size(1))]
else
    --setup MDP
    S = torch.eye(num_state)
    --S = torch.tril(torch.ones(num_state,num_state))
    s_obs = S[s]

    all_statePrime = torch.zeros(num_state*act_dim,num_state)
    all_state = torch.zeros(num_state*act_dim,num_state)
    all_action = torch.zeros(num_state*act_dim,act_dim)
    for s=1,num_state do
        for a=1,act_dim do
            all_state[act_dim*(s-1)+a] = S[s]
            all_action[act_dim*(s-1)+a] = A[a]
            all_statePrime[act_dim*(s-1)+a] = S[T[s][a]]
        end
    end
end
--require 'train_sa_GAN.lua'
--require 'train_policy_GAN.lua'
--require 'train_distinguish.lua'
--require 'train_NCE.lua'
--require 'train_VAE_GAN.lua'
--require 'train_pred_err.lua'
require 'train_pred_GAN.lua'
setup()
softmax = nn.SoftMax()

local num_steps = 1e5
local cumloss =0 

if use_qnet then
    local hid_dim = 100
    local input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
    local output =nn.Linear(hid_dim,act_dim)(hid)
    q_network = nn.gModule({input},{output})
    q_network = q_network:cuda()
    q_w,q_dw = q_network:getParameters()
    q_config = {
        learningRate  = 1e-3
        }
    if use_target_network then
        --target network
        local input = nn.Identity()()
        local hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
        local output =nn.Linear(hid_dim,act_dim)(hid)
        target_network = nn.gModule({input},{output})
        target_network = target_network:cuda()
        target_w,target_dw = target_network:getParameters()
        target_w:copy(q_w)

        target_refresh = 1e3
    end
    mse_crit = nn.MSECriterion():cuda()
end



Q = torch.zeros(num_state,act_dim)
--Q = torch.rand(num_state,act_dim):mul(-1)
print(T)





epsilon = .1
alpha = .1
gamma = .9
net_reward = 0
refresh = 5e2
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
neg_entropy_over_time_correct = torch.zeros(num_steps/refresh,num_state)
neg_entropy_over_time_incorrect = torch.zeros(num_steps/refresh,num_state)
sa_visits = torch.zeros(num_state,act_dim)
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
        if noise_mag > 0 then 
            action_data[i] = torch.rand(act_dim):mul(noise_mag) 
            action_data[i][D.a[mb_ind[i] ] ] = 1 - torch.rand(1):mul(noise_mag)[1]
            --action_data[i] = A[D.a[mb_ind[i] ] ] + torch.rand(act_dim):mul(noise_mag)
            --action_data[i] = action_data[i]:div(action_data[i]:max()) 
        else
            action_data[i] = A[D.a[mb_ind[i] ] ]
        end
        
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
        --a = distributions.cat.rnd(1,softmax:forward(Q[s]))
        if use_egreedy then
            if torch.rand(1)[1] < .1 then
                a[1] = torch.random(act_dim)
            end
        end
    end
    a = a[1]

    --perform action
    sPrime = T[s][a]
    if use_mnist then
        sPrime_obs = digit[sPrime][torch.random(digit[sPrime]:size(1))]
    else
        sPrime_obs = S[sPrime]
    end

    sa_visits[s][a] = sa_visits[s][a] + 1
    visits[sPrime] = visits[sPrime] + 1

    if sPrime == num_state then
        r = 1
        net_reward = net_reward + r
    end
    --[[shaping: 
    r =  gamma^(num_state-sPrime)
    if sPrime > s then
        r = 1 
    else 
        r = 0
    end
    --]]
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
        local action = torch.rand(mb_dim*act_dim,act_dim):mul(noise_mag)
        for i =1,mb_dim do
            mask[mb_ind[i] ] = 1
            for a = 1,act_dim  do
                if noise_mag > 0 then
                    action[mb_dim*(a-1)+i][a] = 1-torch.rand(1):mul(noise_mag)[1]
                    --[[
                    action[mb_dim*(a-1)+i] = A[D.a[mb_ind[i] ] ] + torch.rand(act_dim):mul(noise_mag)
                    action[mb_dim*(a-1)+i] = action[mb_dim*(a-1)+i]:div(action[mb_dim*(a-1)+i]:max()) 
                    --]]
                else
                    action[mb_dim*(a-1)+i][a] = 1
                end
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
        --
        for i=1,mb_dim do
            for a=1,act_dim do
                state[mb_dim*(a-1)+i] = D.obs[mind[i][1] ]
            end
            statePrime[i] = D.obsPrime[mind[i][1] ]
        end
        --]]
        --state = D.obs[mask:expandAs(D.obs)]:reshape(mb_dim,in_dim):repeatTensor(act_dim,1)
        --statePrime = D.obsPrime[mask:expandAs(D.obsPrime)]:reshape(mb_dim,in_dim):repeatTensor(act_dim,1)
        if use_qnet then
            if use_target_network then
                _,qind = q_network:forward(statePrime:cuda()):max(2)
                qPrime = target_network:forward(statePrime:cuda()):gather(2,qind)
                --qPrime,qind = target_network:forward(statePrime:cuda()):max(2)
            else
                qPrime,qind = q_network:forward(statePrime:cuda()):max(2)
            end
        else
            Q_clone = Q:clone()
        end

        --network:evaluate()
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
                --local chance_unknown = (1- C[ind])^5
                local unknown, chance_unknown = get_knownness(C,ind)
                --chance_unknown = torch.rand(1)[1]
                neg_entropy[s[i] ][a] = neg_entropy[s[i] ][a] + chance_unknown

                if  unknown then
                    if a == a_actual[i] then--D.a[mb_ind[i] ] then
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
                Q = q_network:forward(S:cuda())
            end

            gnuplot.raw("set title 'Q-values' ")
            gnuplot.raw('set xrange [' .. .5 .. ':' .. num_state+.5 .. '] noreverse')
            gnuplot.raw('set yrange [*:*] noreverse')
            gnuplot.plot({Q:mean(2)},{Q:max(2):double()},{Q:min(2):double()})
            --gnuplot.imagesc(Q)
        else
            --
            gnuplot.raw("set title 'Q-values' ")
            gnuplot.raw('set xrange [' .. .5 .. ':' .. num_state+.5 .. '] noreverse')
            gnuplot.raw('set yrange [0:1] noreverse')
            gnuplot.plot({Q:mean(2)},{Q:max(2):double()},{Q:min(2):double()})
            --]]
            --gnuplot.imagesc(Q)
        end


        
        gnuplot.raw("set title '% R-Max bonus over time' ")
        gnuplot.raw('set xrange [' .. 0 .. ':' .. t/refresh+1 .. '] noreverse')
        gnuplot.raw('set yrange [0:.1] noreverse')
        local bonus = neg_entropy:cdiv(hist_total+1)
        --gnuplot.plot({bonus:mean(2)},{bonus:max(2):double()},{bonus:min(2):double()}) --dont divide by zero
        print(bonus:mean())
        bonus_hist[t/refresh] = bonus:mean()
        gnuplot.plot(bonus_hist[{{1,t/refresh}}])
        
        
        
        neg_entropy_over_time[t/refresh] = bonus
        neg_entropy_over_time_per_state[t/refresh] = bonus:mean(2)
        neg_entropy_over_time_norm[t/refresh] = bonus:div(bonus:max())
        neg_entropy_over_time_per_state_norm[t/refresh] = bonus:mean(2):div(bonus:mean(2):max())
        for i = 1,num_state do
            neg_entropy_over_time_correct[t/refresh][i] = bonus[i][correct[i] ]/bonus[i]:sum()
            neg_entropy_over_time_incorrect[t/refresh][i] = bonus[i][1]
        end
        gnuplot.raw("set title 'bonus % over time' ")
        gnuplot.imagesc(neg_entropy_over_time[{{1,t/refresh}}])
        gnuplot.raw("set title 'bonus % over time per state' ")
        gnuplot.imagesc(neg_entropy_over_time_per_state[{{1,t/refresh}}])
        gnuplot.raw("set title 'bonus % over time normed' ")
        gnuplot.imagesc(neg_entropy_over_time_norm[{{1,t/refresh}}])
        gnuplot.raw("set title 'bonus % over time per state normed' ")
        gnuplot.imagesc(neg_entropy_over_time_per_state_norm[{{1,t/refresh}}])
        --[[
        --]]
        print(visits)
        visits_over_time[t/refresh] = visits

        
        --[[
        if last_compare then
        gnuplot.raw("set title 'descriminator judgements' ")
        gnuplot.raw('set xrange [' .. .5 .. ':' .. mb_dim+.5 .. '] noreverse')
        gnuplot.raw('set yrange [0:1] noreverse')
        gnuplot.bar(last_compare)
        --gnuplot.axis{0,mb_dim,0,1}
        end
        --]]
        --[[
        gnuplot.raw("set title 'action generation' ")
        sorted,_ = action_data:sort()
        gnuplot.imagesc(sorted)
        print(D.a[D.a:ne(0)]:histc(act_dim):div(D.a:ne(0):sum()))
        if last_compare then
            print(gen_network.output:sum(1):div(gen_network.output:sum())[1])
            print(gen_network.output[1])
        end
        gnuplot.raw("set title 'visits over time' ")
        gnuplot.imagesc(visits_over_time[{{1,t/refresh}}])
        --]]

        cur_known = torch.zeros(num_state,act_dim)
        cur_pred = torch.zeros(num_state*act_dim,in_dim)
        cur_actual_pred = torch.zeros(num_state*act_dim,in_dim)
        cur_err = torch.zeros(num_state,act_dim)
        cur_actual_err = torch.zeros(num_state,act_dim)
        --[[network:evaluate() 
        for s = 1,num_state do
            local state = torch.zeros(1,in_dim)
            if use_mnist then
                state[1] = digit[s][torch.random(digit[s]:size(1))] 
            else
                state[1] = S[s] 
            end
            for a=1,act_dim do
                local sPrime = T[s][a]
                local statePrime = torch.zeros(in_dim)
                if use_mnist then
                    statePrime = digit[sPrime][torch.random(digit[sPrime]:size(1))] 
                else
                    statePrime = S[sPrime] 
                end
                
                local action = torch.rand(1,act_dim):mul(noise_mag)
                action[{{},a}] = -torch.rand(1):mul(noise_mag)+1
                local out = network:forward{state:cuda(),action:cuda()}
                _,chance_unknown = get_knownness(out,1)
                cur_known[s][a] = chance_unknown

                if pred_network then
                --cur_actual_err[s][a] = torch.pow(pred_network.output[1]:double()-statePrime,2):sum()
                cur_actual_err[s][a] = BCE(pred_network.output:double(),statePrime:reshape(1,in_dim))
                cur_pred[act_dim*(s-1)+a] = pred_network.output[1]:double():clone()
                cur_actual_pred[act_dim*(s-1)+a] = statePrime:clone()
                cur_err[s][a] = err_network.output[1][1]
                end
            end
        end
        network:training()
        --]]

        network:forward{all_state:cuda(),all_action:cuda()}
        cur_pred = pred_network.output:double()
        cur_actual_pred = all_statePrime
        cur_err = get_knownness(err_network.output):reshape(num_state,act_dim)
        cur_actual_err = BCE(cur_pred,cur_actual_pred):reshape(num_state,act_dim)
        
        --gnuplot.raw("set title 'current D estimates' ")
        --gnuplot.imagesc(cur_known)
        gnuplot.raw("set title 'current pred' ")
        gnuplot.imagesc(cur_pred)
        gnuplot.raw("set title 'pred diff' ")
        gnuplot.imagesc(cur_actual_pred - cur_pred)
        gnuplot.raw("set title 'current pred pred err' ")
        gnuplot.imagesc(cur_err)
        gnuplot.raw("set title 'current actual pred err' ")
        gnuplot.imagesc(cur_actual_err)
        --[[
        gnuplot.raw("set title 'visits' ")
        gnuplot.imagesc(sa_visits:log1p())
        --]]
        --[[
        gnuplot.raw("set title 'actions' ")
        gnuplot.imagesc(action_data)
        --]]
        --[[
        gnuplot.raw("set title 'visits over time' ")
        gnuplot.imagesc(visits_over_time[{{1,t/refresh}}]:log1p())
        --]]
        --gnuplot.raw("set title 'bonus % over time for correct action' ")
        --gnuplot.imagesc(neg_entropy_over_time_correct[{{1,t/refresh}}])
        --gnuplot.raw("set title 'bonus % over time for incorrect action' ")
        --gnuplot.imagesc(neg_entropy_over_time_incorrect[{{1,t/refresh}}])

        print(action_data[1])
        print(action_data[-1])
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




