require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
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

local num_steps = 1e5
local cumloss =0 




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
epsilon = .1
alpha = .1
gamma = .9
net_reward = 0
refresh = 1e2
bonus_hist = torch.zeros(num_steps/refresh)
C = torch.zeros(num_state,act_dim)
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
D = {}
D.size = 1e4
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.r = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.digit = torch.zeros(D.size,digit[1]:size(2))--sPrime digit
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
for t=1,num_steps do
    r = 0
    --select action
    _,a = torch.max(Q[s],1)
    a = a[1]

    --perform action
    sPrime = T[s][a]
    visits[sPrime] = visits[sPrime] + 1

    if sPrime == num_state then
        r = 1
        net_reward = net_reward + r
    end
    --record history
    D.s[D.i] = s
    D.a[D.i] = a
    D.r[D.i] = r
    D.sPrime[D.i] = sPrime
    D.digit[D.i] = digit[sPrime][torch.random(digit[sPrime]:size(1))] 
    D.i = (D.i % D.size) + 1

    --update model params
    if t > mb_dim then
        --gotta re-perm if runnning train_dis multiple times
        mb_ind = torch.randperm(math.min(t,D.size))
        --update adver nets
        x,batchloss = optim.adam(train,w,config)
        cumloss = cumloss + batchloss[1]
        --update Q
        for i = 1,mb_dim do
            local s,a,r,sPrime
            s = D.s[mb_ind[i]]
            for a = 1,act_dim do
                network:evaluate()
                local action = torch.rand(1,act_dim):mul(noise_mag)
                action[1][a] = 1-torch.rand(1):mul(noise_mag)[1]
                local possible
                if use_gpu then
                    possible = {D.digit[mb_ind[i] ]:reshape(1,in_dim):cuda(),action:cuda()}
                else
                    possible = {D.digit[mb_ind[i] ],action}
                end
                C[s][a] = network:forward(possible)[1][1]
            end
            --you can experience all actions under threshold, since they all go to heaven!
            local known_flag = true
            for j = 1,act_dim do
                a = j
                hist_total[s][a] = hist_total[s][a] + 1
                local chance_unknown = (1 - H(C[s][a]))^(1/temp)
                neg_entropy[s][a] = neg_entropy[s][a] + chance_unknown
                if  chance_unknown > torch.rand(1)[1] then
                    if a == D.a[mb_ind[i] ] then
                        known_flag = false
                    end
                    sPrime = s
                    r = 1
                    Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*r
                end
            end
            --only experienced actions can be updated over threshold
            a = D.a[mb_ind[i]]
            if known_flag then
                --print('known!')
                r = D.r[mb_ind[i]]
                sPrime = D.sPrime[mb_ind[i]]
                if s == num_state then
                    Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*r
                else
                    Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
                end
            end
        end
    end
    s = sPrime

    if t % refresh == 0 then
        gnuplot.figure(1)
        gnuplot.raw("set multiplot layout 2,4 columnsfirst")


        gnuplot.raw("set title 'Q-values' ")
        gnuplot.raw('set xrange [' .. .5 .. ':' .. num_state+.5 .. '] noreverse')
        gnuplot.raw('set yrange [0:1] noreverse')
        gnuplot.plot({Q:mean(2)},{Q:max(2):double()},{Q:min(2):double()})


        
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

        gnuplot.raw("set title 'action generation' ")
        sorted,_ = action_data:sort()
        gnuplot.imagesc(sorted)
        print(D.a[D.a:ne(0)]:histc(act_dim))
        if last_compare then
            print(gen_network.output:sum(1))
        end

        gnuplot.raw("set title 'visits over time' ")
        gnuplot.imagesc(visits_over_time[{{1,t/refresh}}])

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




