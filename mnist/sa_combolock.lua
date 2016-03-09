require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
torch.setnumthreads(1)

digit = torch.load('digit.t7')
in_dim = digit[1]:size(2)

hid_dim = 100
out_dim = 1
dropout_p = .1
--rev_grad = true
--Discrim
local input = nn.Identity()()
local hid_lin = nn.Linear(in_dim,hid_dim)
local hid = nn.Dropout(dropout_p)(nn.ReLU()(hid_lin(input)))
local out_lin = nn.Linear(hid_dim,out_dim)
local output = nn.Sigmoid()(out_lin(hid))
network = nn.gModule({input},{output})
--Gen
noise_dim = 20
gen_hid_dim = 100
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(hid))
gen_network = nn.gModule({input},{output})

--full
local full_input = nn.Identity()()
local connect
if rev_grad then
    connect= nn.GradientReversal()(gen_network(full_input))
else
    connect= gen_network(full_input)
end
local full_out = network(connect)
full_network = nn.gModule({full_input},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = 320
local data = torch.zeros(mb_dim,in_dim)
local target = torch.zeros(mb_dim,1)
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)
gen_count = 0
local train_dis = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:training()
    
    for i=1,mb_dim/2 do
        data[i] = D.digit[mb_ind[i]]
    end
    target[{{1,mb_dim/2}}] = torch.ones(mb_dim/2)

    noise_data = torch.randn(mb_dim/2,noise_dim)
    target[{{mb_dim/2+1,-1}}] = torch.zeros(mb_dim/2)
    data[{{mb_dim/2+1,-1}}]  = gen_network:forward(noise_data)

    output = network:forward(data)
    thresh = output[mb_dim/2+1][1] --sample threshold
    --print(output[{{1,mb_dim/2}}]:mean(),output[{{mb_dim/2+1,-1}}]:mean())
    gen_qual = (gen_qual*gen_count + output[{{mb_dim/2+1,-1}}]:mean())/(gen_count+1)
    gen_count = gen_count + 1
    thresh = gen_qual
    thresh = .05
    loss = bce_crit:forward(output,target)
    grad = bce_crit:backward(output,target)
    network:backward(data,grad)
    return loss,dw
end
local train_gen = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:evaluate()

    local noise_data = torch.randn(mb_dim,noise_dim)
    if rev_grad then
        target:zero()
    else
        target:zero():add(1)
    end
    local output = full_network:forward(noise_data)
    local loss = bce_crit:forward(output,target)
    local grad = bce_crit:backward(output,target)
    full_network:backward(noise_data,grad)
    return loss,dw
end
config_dis = {
    learningRate  = 1e-3
    }
config_gen = {
    learningRate  = 1e-3
    }
local num_steps = 1e6
local cumloss =0 




num_state = 10
Q = torch.zeros(num_state,4)
T = torch.ones(num_state,4)
correct = torch.ones(num_state)
for i = 1,num_state-1 do
    action = torch.random(4)
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
C = torch.zeros(num_state,4)
hist_C = torch.zeros(refresh,num_state)
true_C = torch.zeros(num_state)
thresh = .4
thresh_dif = .4
hist_thresh = torch.zeros(num_state,4)
hist_known = torch.zeros(num_state,4)--,refresh*mb_dim*4)
hist_total = torch.zeros(num_state,4)
gen_qual = 0
D = {}
D.size = 1e4
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.r = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.digit = torch.zeros(D.size,in_dim)--sPrime digit
D.i = 1
local plot1 = gnuplot.figure()
gnuplot.axis{.5,num_state+.5,0,1}
local plot2 = gnuplot.figure()
local plot3 = gnuplot.figure()
gnuplot.axis{.5,num_state+.5,-.1,1.1}
for t=1,num_steps do
    r = 0
    --select action
    if rmax then
        _,a = torch.max(Q[s],1)
        a = a[1]
    elseif torch.rand(1)[1] < epsilon then
        a = torch.random(4)
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
        --update adver nets
        for k=1,5 do
            mb_ind = torch.randperm(math.min(t,D.size))
            x,batchloss = optim.rmsprop(train_dis,w,config_dis)
        end
        x,batchloss = optim.rmsprop(train_gen,w,config_gen)
        cumloss = cumloss + batchloss[1]
        --update Q
        for i = 1,mb_dim do
            local s,a,r,sPrime
            s = D.s[mb_ind[i]]
            if rmax then
                for a = 1,4 do
                    local digit_id = T[s][a]
                    local sample_ind = torch.random(digit[digit_id]:size(1))
                    local digit_sample = digit[digit_id][sample_ind]
                    network:evaluate()
                    C[s][a] = network:forward(digit_sample)[1] --cheating
                end
                hist_C[(t-1) % refresh +1] = C:mean(2)
                --you can experience all actions under threshold, since they all go to heaven!
                for j = 1,4 do
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
        gnuplot.imagesc(gen_network:forward(torch.randn(noise_dim)):reshape(28,28))


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




