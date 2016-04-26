require 'nngraph'
require 'optim'
require 'gnuplot'
--require 'cunn'
require 'Bootstrap'
num_state = 20
in_dim = num_state
hid_dim = 100
act_dim = 4
heads = 10
mb_dim = 32
gamma = .9
input = nn.Identity()()
hid1 = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
--hid2 = nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid1))
last_hid = hid1
sample_head = nn.Linear(hid_dim,act_dim)
bootstrap = nn.Bootstrap(sample_head,heads)
output = bootstrap(last_hid)
network = nn.gModule({input},{output})
w,dw = network:getParameters()

--setup MDP
A = torch.eye(act_dim)
S = torch.eye(num_state)
T = torch.Tensor(num_state,act_dim)
correct = torch.Tensor(num_state)
for s = 1,num_state do
    correct[s] = torch.random(act_dim)
    for a =1,act_dim do
        if a==correct[s] and s < num_state then
            T[s][a] = s+1
        else
            T[s][a] = 1
        end
    end
end

D = {}
D.size = 1e4
D.s = torch.Tensor(D.size)
D.a = torch.Tensor(D.size)
D.r = torch.Tensor(D.size)
D.sPrime = torch.Tensor(D.size)
D.i = 0

crit = nn.MSECriterion()
--pseudo arguments:
--data
--target
--aind
train = function(w)
    network:zeroGradParameters()
    save = bootstrap.active
    bootstrap.active = {}
    output = network:forward(data)
    used = output:gather(2,aind)
    loss = crit:forward(used,target)
    grad = crit:backward(used,target)
    masked_grad = torch.zeros(output:size()):scatter(2,aind,grad)
    network:backward(data,masked_grad)
    bootstrap.active = save
    return loss,dw
end
refresh = 1e3
cumloss = 0
config = {learningRate=1e-3}
s = 1
visits = torch.zeros(num_state)
for t=1,1e6 do
    visits[s] = visits[s] + 1
    --select action
    q = network:forward(S[s])
    _,a = q:max(1)
    a = a[1]
    --get results
    sPrime = T[s][a]
    if sPrime == num_state then
        r = 1
    else
        r = 0
    end
    --record history
    D.i = (D.i % D.size) + 1
    D.s[D.i] = s
    D.a[D.i] = a
    D.r[D.i] = r
    D.sPrime[D.i] = sPrime

    --update current state
    s = sPrime
    if s == 1 then
        --new 'episode'
        bootstrap.active = {}
    end

    --train
    if t > mb_dim then
        --sample minibatch
        mb_ind = torch.randperm(math.min(t,D.size))
        data = torch.Tensor(mb_dim,in_dim)
        dataReward = torch.Tensor(mb_dim)
        dataPrime = torch.Tensor(mb_dim,in_dim)
        aind = torch.zeros(mb_dim,1):long()
        for i =1,mb_dim do
            data[i] = S[ D.s[mb_ind[i] ] ]
            aind[i] = D.a[mb_ind[i] ] 
            dataReward[i] = D.r[mb_ind[i] ] 
            dataPrime[i] = S[ D.sPrime[mb_ind[i] ] ]
        end
        qPrime = network:forward(dataPrime)
        target = dataReward + qPrime:max(2)*gamma
        _,batchloss = optim.adam(train,w,config)
        cumloss = cumloss + batchloss[1]
        if t %refresh == 0 then
            print(t,cumloss,w:norm(),dw:norm())
            print(visits)
            visits:zero()
            cumloss = 0 
        end
    end
end
