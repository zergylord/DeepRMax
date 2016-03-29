require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
require 'cunn'
--torch.manualSeed(123)
--cutorch.manualSeed(123)
torch.setnumthreads(1)
log2 = function(x) return torch.log(x)/torch.log(2) end
--H = function(p) return log2(p):cmul(-p)-log2(-p+1):cmul(-p+1) end
H = function(p) return log2(p)*(-p)-log2(-p+1)*(-p+1) end
noise_mag = .05
thresh = .92 --.2 --.04
temp = .5
use_action = true

act_dim = 4

local timer = torch.Timer()

digit = torch.load('digit.t7')

--require 'train_sa_GAN.lua'
require 'train_policy_GAN.lua'
softmax = nn.SoftMax():cuda()

local num_steps = 1e5
local cumloss =0 

--use_qnet = true
if use_qnet then
    local hid_dim = 100
    local input = nn.Identity():cuda()()
    --local hid = nn.BatchNormalization(hid_dim):cuda()(nn.ReLU():cuda()(nn.Linear(in_dim,hid_dim):cuda()(input)))
    local hid = nn.ReLU():cuda()(nn.Linear(in_dim,hid_dim):cuda()(input))
    local output =nn.Linear(hid_dim,act_dim):cuda()(hid)
    q_network = nn.gModule({input},{output})
    q_w,q_dw = q_network:getParameters()
    q_config = {
        learningRate  = 1e-4
        }
        mse_crit = nn.MSECriterion():cuda()
end



num_state = 10
Q = torch.zeros(num_state,act_dim)
--Q = torch.rand(num_state,act_dim):mul(-1)
T = torch.ones(num_state,act_dim)
correct = torch.ones(num_state)
for i = 1,num_state-1 do
    action = torch.random(act_dim)
    T[i][action] = i+1
    correct[i] = action
end





s = 1
s_digit = digit[s][torch.random(digit[s]:size(1))]
epsilon = .1
alpha = .1
gamma = .9
net_reward = 0
refresh = 1e2
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
D.size = 1e4
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.r = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.digit = torch.zeros(D.size,digit[1]:size(2))--s digit
D.digitPrime = torch.zeros(D.size,digit[1]:size(2))--sPrime digit
D.i = 1
local get_data = function(data,action_data)
    num = data:size(1)
    for i=1,num do
        data[i] = D.digit[mb_ind[i]]
        if use_action then
            action_data[i] = torch.rand(act_dim):mul(noise_mag) 
            action_data[i][D.a[mb_ind[i] ] ] = 1 - torch.rand(1):mul(noise_mag)[1]
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
        local vals = q_network:forward(s_digit:view(1,in_dim):cuda())
        _,a = vals:max(2)
        a = a[1]
    else
        _,a = torch.max(Q[s],1)
    end
    a = a[1]

    --perform action
    sPrime = T[s][a]
    sPrime_digit = digit[sPrime][torch.random(digit[sPrime]:size(1))]
    sa_visits[s][a] = sa_visits[s][a] + 1
    visits[sPrime] = visits[sPrime] + 1

    if sPrime == num_state then
        r = 1
        net_reward = net_reward + r
    end
    --shaping: 
    --r = torch.exp(sPrime/num_state)
    --record history
    D.s[D.i] = s
    D.a[D.i] = a
    D.r[D.i] = r
    D.sPrime[D.i] = sPrime
    D.digit[D.i] = s_digit:clone() 
    D.digitPrime[D.i] = sPrime_digit:clone() 
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
                action[mb_dim*(a-1)+i][a] = 1-torch.rand(1):mul(noise_mag)[1]
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
                state[mb_dim*(a-1)+i] = D.digit[mind[i][1] ]
            end
            statePrime[i] = D.digitPrime[mind[i][1] ]
        end
        --]]
        --state = D.digit[mask:expandAs(D.digit)]:reshape(mb_dim,in_dim):repeatTensor(act_dim,1)
        --statePrime = D.digitPrime[mask:expandAs(D.digitPrime)]:reshape(mb_dim,in_dim):repeatTensor(act_dim,1)
        if use_qnet then
            qPrime,qind = q_network:forward(statePrime:cuda()):max(2)
        else
            Q_clone = Q:clone()
        end

        network:evaluate()
        local possible
        if use_gpu then
            possible = {state:cuda(),action:cuda()}
        else
            possible = {state,action}
        end
        C = network:forward(possible):squeeze()
        target = torch.zeros(mb_dim*act_dim)
        target_mask = torch.zeros(mb_dim*act_dim,1):byte()
        for i=1,mb_dim do
            --you can experience all actions under threshold, since they all go to heaven!
            local known_flag = true
            for a = 1,act_dim do
                local ind = mb_dim*(a-1)+i
                hist_total[s[i] ][a] = hist_total[s[i] ][a] + 1
                local chance_unknown = (1 - H(C[ind]))^(1/temp)
                neg_entropy[s[i] ][a] = neg_entropy[s[i] ][a] + chance_unknown
                if  chance_unknown > torch.rand(1)[1] then
                    if a == a_actual[i] then--D.a[mb_ind[i] ] then
                        known_flag = false
                    end
                    local r = 1
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
    s_digit = sPrime_digit:clone() 
    if t % refresh == 0 then
        gnuplot.figure(1)
        gnuplot.raw("set multiplot layout 2,4 columnsfirst")


        if use_qnet then
            local data = torch.zeros(num_state,in_dim):cuda()
            for s = 1,num_state do
                data[s] = digit[s][torch.random(digit[s]:size(1))]
            end
            local Q = q_network:forward(data)
            gnuplot.raw("set title 'Q-values' ")
            gnuplot.raw('set xrange [' .. .5 .. ':' .. num_state+.5 .. '] noreverse')
            gnuplot.raw('set yrange [-1:1] noreverse')
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
            neg_entropy_over_time_correct[t/refresh][i] = bonus[i][correct[i] ]
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
        gnuplot.raw("set title 'bonus % over time for correct action' ")
        gnuplot.imagesc(neg_entropy_over_time_correct[{{1,t/refresh}}])
        gnuplot.raw("set title 'bonus % over time for incorrect action' ")
        gnuplot.imagesc(neg_entropy_over_time_incorrect[{{1,t/refresh}}])
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
        network:evaluate() 
        local iter = 10
        for s = 1,num_state do
            local state = torch.zeros(iter,digit[s]:size(2))
            for i=1,iter do
                state[i] = digit[s][torch.random(digit[s]:size(1))] 
            end
            for a=1,act_dim do
                local action = torch.rand(iter,act_dim):mul(noise_mag)
                action[{{},a}] = -torch.rand(iter):mul(noise_mag)+1
                local out = network:forward{state:cuda(),action:cuda()}:double()
                for i=1,iter do
                    cur_known[s][a] = cur_known[s][a] + ((1- H(out[i]))^(1/temp)) /iter
                end
            end
        end
        network:training()
        
        gnuplot.raw("set title 'current D estimates' ")
        gnuplot.imagesc(cur_known)
        gnuplot.raw("set title 'total visits' ")
        gnuplot.imagesc(sa_visits:log1p())
        


        gnuplot.raw('unset multiplot')

        print(thresh,t,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        gen_count = 0
        timer:reset()
        cumloss = 0
        if net_reward/refresh > ((1/num_state)*(1-epsilon)) then
            final_time = t
            break
        end
        net_reward = 0
        visits:zero()
        hist_total:zero()
        neg_entropy:zero()
        collectgarbage()
    end

end




