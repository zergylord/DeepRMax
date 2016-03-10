require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
torch.setnumthreads(1)
--afterstate = true

local timer = torch.Timer()

digit = torch.load('digit.t7')

require 'train_mnist.lua'

local num_steps = 1e6
local cumloss =0 




num_state = 10
act_dim = 4
Q = torch.zeros(num_state,act_dim)
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
refresh = 1e3
rmax = true
C = torch.zeros(num_state,act_dim)
hist_C = torch.zeros(refresh,num_state)
true_C = torch.zeros(num_state)
thresh = .4
thresh_dif = .4
hist_thresh = torch.zeros(num_state,act_dim)
hist_known = torch.zeros(num_state,act_dim)--,refresh*mb_dim*act_dim)
hist_total = torch.zeros(num_state,act_dim)
gen_qual = 0
D = {}
D.size = 1e4
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.r = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.digit = torch.zeros(D.size,digit[1]:size(2))--sPrime digit
D.i = 1
local plot1 = gnuplot.figure()
gnuplot.axis{.5,num_state+.5,0,1}
local plot2 = gnuplot.figure()
local plot3 = gnuplot.figure()
local plot4 = gnuplot.figure()
gnuplot.axis{.5,num_state+.5,-.1,1.1}

local get_data = function(data)
    for i=1,mb_dim/2 do
        if afterstate then
            data[i] = D.digit[mb_ind[i]]
        else
            local action = torch.zeros(act_dim) 
            action[D.a[mb_ind[i] ] ] = 1
            data[i] = D.digit[mb_ind[i] ]:cat(action)
        end
    end
end
set_data_func(get_data)
for t=1,num_steps do
    r = 0
    --select action
    if rmax then
        _,a = torch.max(Q[s],1)
        a = a[1]
    elseif torch.rand(1)[1] < epsilon then
        a = torch.random(act_dim)
    else
        _,a = torch.max(Q[s],1)
        a = a[1]
        a = correct[s]
    end

    --perform action
    sPrime = T[s][a]

    --network does this
    --true_C[s][a] = true_C[s][a] + 1
    true_C[sPrime] = true_C[sPrime] + 1
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
            if rmax then
                for a = 1,act_dim do
                    if afterstate then
                        local digit_id = T[s][a]
                        local sample_ind = torch.random(digit[digit_id]:size(1))
                        local digit_sample = digit[digit_id][sample_ind]
                        network:evaluate()
                        C[s][a] = network:forward(digit_sample)[1] --cheating
                    else
                        local action = torch.zeros(act_dim) 
                        action[D.a[mb_ind[i] ] ] = 1
                        local possible = D.digit[mb_ind[i] ]:cat(action)
                        C[s][a] = network:forward(possible)[1] --cheating
                    end
                end
                hist_C[(t-1) % refresh +1] = C:mean(2)
                --you can experience all actions under threshold, since they all go to heaven!
                for j = 1,act_dim do
                    a = j
                    hist_total[s][a] = hist_total[s][a] + 1
                    --local visits = hist_total:sum(2)[s][1]
                    --hist_known[s][visits] =  C[s][a]
                    hist_known[s][a] = hist_known[s][a] + C[s][a]
                    --if C[s][a] < (thresh-thresh_dif) or C[s][a] > (thresh+thresh_dif) then
                    --if C[s][a] > thresh then
                    if C[s][a] < thresh then
                        hist_thresh[s][a] = hist_thresh[s][a] + 1
                        sPrime = s
                        --r = (1-gamma)*1
                        --Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
                        r = 1
                        Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*r
                    end
                end
                --only experienced actions can be updated over threshold
                a = D.a[mb_ind[i]]
                --if C[s][a] >= (thresh-thresh_dif) and C[s][a] <= (thresh+thresh_dif) then
                --if C[s][a] <= thresh then
                if C[s][a] >= thresh then
                    --print('known!')
                    r = D.r[mb_ind[i]]
                    sPrime = D.sPrime[mb_ind[i]]
                    if s == num_state then
                        Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*r
                    else
                        Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
                    end
                end
            else
                r = D.r[mb_ind[i]]
                sPrime = D.sPrime[mb_ind[i]]
                a = D.a[mb_ind[i]]
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
        gnuplot.figure(plot1)
        --[[Known-ness values
        local avgs = hist_C:mean(1)[{1,{}}]
        local spread = hist_C:std(1)[{1,{}}]
        gnuplot.plot({avgs},{avgs+spread},{avgs-spread})
        hist_C:zero()
        --]]
        --Q-values
        gnuplot.plot({Q:mean(2)},{Q:max(2):double()},{Q:min(2):double()})

        --percent of time a state-action gets R-Max bonus
        gnuplot.figure(plot2)
        --gnuplot.bar(true_C)
        local bonus = hist_thresh:cdiv(hist_total+1)
        gnuplot.plot({bonus:mean(2)},{bonus:max(2):double()},{bonus:min(2):double()}) --dont divide by zero
        hist_thresh:zero()
        --true_C:zero()
        
        gnuplot.figure(plot3)
        local known = hist_known:cdiv(hist_total+1)
        gnuplot.plot({known:mean(2)},{known:max(2):double()},{known:min(2):double()},{torch.ones(num_state):mul(gen_qual)})
        print(hist_total)

        hist_known:zero()
        hist_total:zero()
        
        gnuplot.figure(plot4)
        --gnuplot.imagesc(gen_network:forward(torch.randn(noise_dim)):reshape(28,28))
        if afterstate then
            gnuplot.imagesc(data[{{mb_dim/2+1}}]:reshape(28,28))
        else
            gnuplot.imagesc(data[{{mb_dim/2+1},{1,28*28}}]:reshape(28,28))
            print(data[{{mb_dim/2+1},{28*28+1,-1}}])
        end


        print(thresh,t,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
        gen_qual = 0
        gen_count = 0
        timer:reset()
        cumloss = 0
        if net_reward/refresh > ((1/num_state)*(1-epsilon)) then
            break
        end
        net_reward = 0
        collectgarbage()
    end

end




