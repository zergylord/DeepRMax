require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'hdf5'
torch.setnumthreads(4)
f = hdf5.open('mnist.hdf5')
mnist_data = f:read():all()
digit = {}
in_dim = mnist_data.x_train:size(2)
for i=1,10 do
    mask = mnist_data.t_train:ne(i):reshape(50000,1):expandAs(mnist_data.x_train)
    local digit_data = mnist_data.x_train[mask]
    digit[i] = digit_data:reshape(digit_data:size(1)/in_dim,in_dim):double()
end

hid_dim = 500
out_dim = 1
--Discrim
local input = nn.Identity()()
local hid_lin = nn.Linear(in_dim,hid_dim)
local hid = nn.ReLU()(hid_lin(input))
local out_lin = nn.Linear(hid_dim,out_dim)
local output = nn.Sigmoid()(out_lin(hid))
network = nn.gModule({input},{output})
--Gen
noise_dim = 20
gen_hid_dim = 500
local input = nn.Identity()()
local hid = nn.ReLU()(nn.Linear(noise_dim,gen_hid_dim)(input))
local output =nn.Sigmoid()( nn.Linear(gen_hid_dim,in_dim)(hid))
gen_network = nn.gModule({input},{output})

--full
local full_input = nn.Identity()()
local connect= nn.GradientReversal()(gen_network(full_input))
local full_out = network(connect)
full_network = nn.gModule({full_input},{full_out})


w,dw = full_network:getParameters()
local timer = torch.Timer()
local bce_crit = nn.BCECriterion()
local net_reward = 0
local mb_dim = 32
local data = torch.zeros(mb_dim,in_dim)
local target = torch.zeros(mb_dim,1)
local mu = torch.randn(in_dim)
local sigma = torch.rand(in_dim)
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

    print(data:size())
    output = network:forward(data)
    loss = bce_crit:forward(output,target)
    grad = bce_crit:backward(output,target)
    network:backward(data,grad)
    r = (output[{{1,mb_dim/2}}]:sum() + output[{{mb_dim/2+1,-1}}]:mul(-1):add(1):sum() )/mb_dim
    net_reward = net_reward + r
    return loss,dw
end
local train_gen = function(x)
    if x ~= w then
        w:copy(x)
    end
    full_network:zeroGradParameters()
    network:evaluate()

    local noise_data = torch.randn(mb_dim,noise_dim)
    target:zero()
    local output = full_network:forward(noise_data)
    local loss = bce_crit:forward(output,target)
    local grad = bce_crit:backward(output,target)
    full_network:backward(data,grad)
    return loss,dw
end
config = {
    learningRate  = 1e-3
    }
local num_steps = 1e6
local refresh = 1e2
local cumloss =0 




num_state = 100
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
thresh = 1
D = {}
D.size = 1e3
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.r = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.digit = torch.zeros(D.size,in_dim)--sPrime digit
D.i = 1
for t=1,1e6 do
    r = 0
    --select action
    if rmax then
        _,a = torch.max(Q[s],1)
        a = a[1]
        --[[
        for i = 1,4 do
            if C[s][i] < thresh then
               a = i
               --r = (1-gamma)*1
            end
        end
        --]]
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
    --C[s][a] = C[s][a] + 1
    if sPrime == num_state then
        r = 1
        net_reward = net_reward + r
    end
    --record history
    D.s[D.i] = s
    D.a[D.i] = a
    D.r[D.i] = r
    D.sPrime[D.i] = sPrime
    D.digit = digit[sPrime][torch.random(digit[sPrime]:size(1))] 
    D.i = (D.i % D.size) + 1
    --update model params
    if t > mb_dim then
        mb_ind = torch.randperm(math.min(t,D.size))
        --update adver nets
        for k=1,1 do
            x,batchloss = optim.rmsprop(train_dis,w,config)
        end
        x,batchloss = optim.rmsprop(train_gen,w,config)
        cumloss = cumloss + batchloss[1]
        if t %refresh == 0 then
            print(t,net_reward/refresh,cumloss,w:norm(),dw:norm(),timer:time().real)
            timer:reset()
            net_reward = 0
            cumloss = 0
        end
        --update Q
        for i = 1,mb_dim do
            local s,a,r,sPrime
            s = D.s[mb_ind[i]]
            if rmax then
                for a = 1,4 do
                    local digit_id = T[s][a]
                    local sample_ind = torch.random(digit[digit_id]:size(1))
                    local digit_sample = digit[digit_id][sample_ind]
                    C[s][a] = network:forward(digit_sample)[1] --cheating
                end
                --you can experience all actions under threshold, since they all go to heaven!
                for j = 1,4 do
                    a = j
                    if C[s][a] < thresh then
                        r = (1-gamma)*1
                        sPrime = s
                        Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
                    end
                end
                --only experienced actions can be updated over threshold
                a = D.a[mb_ind[i]]
                if C[s][a] >= thresh then
                    r = D.r[mb_ind[i]]
                    sPrime = D.sPrime[mb_ind[i]]
                    Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
                end
            else
                r = D.r[mb_ind[i]]
                sPrime = D.sPrime[mb_ind[i]]
                a = D.a[mb_ind[i]]
                Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
            end
        end
    end
    s = sPrime

    if t % refresh == 0 then
        print(t,net_reward/refresh)
        gnuplot.plot(C:sum(2))
        print(Q)
        if net_reward/refresh > ((1/num_state)*(1-epsilon)) then
            break
        end
        net_reward = 0
    end

end




