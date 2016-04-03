require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
require 'LinearVA'
require 'KLDCriterion'
require 'Reparametrize'
state_dim = 10
act_dim = 4
fact_dim = 15
hid_dim = 20
dist_dim = 2
config = {learningRate = 1e-3}
kl_scale = 1e-5

--setup network
input = nn.Identity()()
action = nn.Identity()()
--last_hid = nn.ReLU()(nn.Linear(fact_dim,hid_dim)(nn.CMulTable(){nn.Linear(state_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)}))
--
factor = nn.CMulTable(){nn.Linear(state_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)}
mu = nn.LinearVA(fact_dim,dist_dim)(factor)
sigma = nn.LinearVA(fact_dim,dist_dim)(factor)
encoder = nn.gModule({input,action},{mu,sigma})

input = nn.Identity()()
action = nn.Identity()()
last_hid = nn.Reparametrize(dist_dim)(encoder{input,action})
--last_hid = nn.ReLU()(nn.Linear(hid_dim,hid_dim)(hid))
--]]

output = nn.Sigmoid()(nn.Linear(dist_dim,state_dim)(last_hid))
network = nn.gModule({input,action},{output})
w,dw = network:getParameters()
mse_crit = nn.MSECriterion()
kl_crit = nn.KLDCriterion()

--setup MDP
A = torch.eye(act_dim)
S = torch.eye(state_dim)
T = distributions.cat.rnd(act_dim*state_dim,torch.ones(state_dim)):reshape(state_dim,act_dim)
D = {}
D.s = S:repeatTensor(act_dim,1)
D.a = A:repeatTensor(state_dim,1):t():reshape(act_dim*state_dim,act_dim)
D.sPrime = torch.zeros(act_dim*state_dim,state_dim)
for a = 1,act_dim do
    for s = 1,state_dim do
        D.sPrime[state_dim*(a-1)+s] = S[T[s][a] ]
    end
end
mb_dim = act_dim*state_dim
mb_s = torch.zeros(mb_dim,state_dim)
mb_a = torch.zeros(mb_dim,act_dim)
mb_sPrime = torch.zeros(mb_dim,state_dim)
train = function(x)
    network:zeroGradParameters()
    local shuffle = torch.randperm(mb_dim)
    --[[ only needed for minibatches
    for i = 1,mb_dim do
        mb_s[shuffle[i] ]:copy(D.s[shuffle[i] ])
        mb_a[shuffle[i] ]:copy(D.a[shuffle[i] ])
        mb_sPrime[shuffle[i] ]:copy(D.sPrime[shuffle[i] ])
    end
    local o = network:forward{mb_s,mb_a}
    local loss = mse_crit:forward(o,mb_sPrime)
    local grad = mse_crit:backward(o,mb_sPrime)
    network:backward({mb_s,mb_a},grad)
    --]]
    local o = network:forward{D.s,D.a}
    local loss = mse_crit:forward(o,D.sPrime)
    local grad = mse_crit:backward(o,D.sPrime)
    local kl_loss = kl_crit:forward(encoder.output)
    local kl_grad = kl_crit:backward(encoder.output)
    network:backward({D.s,D.a},grad)
    encoder:backward({D.s,D.a},{-kl_grad[1]:mul(kl_scale),-kl_grad[2]:mul(kl_scale)})
    return loss-kl_loss,dw
end

refresh = 5e3
cumloss = 0
for i = 1,1e5 do
    _,batchloss = optim.adam(train,w,config)
    cumloss = cumloss + batchloss[1]
    if i % refresh == 0 then
        print(i,cumloss/refresh,w:norm(),dw:norm())
        gnuplot.raw("set multiplot layout 2,2 columnsfirst")
        --gnuplot.figure(1)
        local o = network:forward{D.s:cat(torch.rand(10,state_dim):gt(.4):double(),1),D.a:cat(torch.rand(10,act_dim):gt(.3):double(),1)}
        gnuplot.imagesc(o)
        --gnuplot.figure(2)
        gnuplot.imagesc(D.sPrime)
        --gnuplot.figure(3)
        gnuplot.imagesc(mu.data.module.output)
        --gnuplot.figure(4)
        gnuplot.imagesc(sigma.data.module.output)
        gnuplot.raw('unset multiplot')
        cumloss = 0
    end
end
