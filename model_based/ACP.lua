require 'nngraph'
require 'optim'
require 'distributions'
require 'gnuplot'
state_dim = 10
act_dim = 4
fact_dim = 15
hid_dim = 20
config = {learningRate = 1e-3}

--setup network
input = nn.Identity()()
action = nn.Identity()()
factor = nn.ReLU()(nn.Linear(fact_dim,hid_dim)(nn.CMulTable(){nn.Linear(state_dim,fact_dim)(input),nn.Linear(act_dim,fact_dim)(action)}))
output = nn.Sigmoid()(nn.Linear(hid_dim,state_dim)(factor))
network = nn.gModule({input,action},{output})

--error pred network
input = nn.Identity()()
action = nn.Identity()()
pred = nn.Identity()()
hid = nn.ReLU()(nn.CAddTable(){nn.Linear(state_dim,hid_dim)(input),nn.Linear(act_dim,hid_dim)(action),nn.Linear(state_dim,hid_dim)(pred)})
output = nn.Linear(hid_dim,1)(hid)
err_network = nn.gModule({input,action,pred},{output})

--full network (mainly for gathering params)
input = nn.Identity()()
action = nn.Identity()()
output = err_network{input,action,network{input,action}}
full_network = nn.gModule({input,action},{output})

w,dw = full_network:getParameters()
mse_crit = nn.MSECriterion()

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

train = function(x)
    full_network:zeroGradParameters()
    local o = network:forward{D.s,D.a}
    local loss = mse_crit:forward(o,D.sPrime)
    local grad = mse_crit:backward(o,D.sPrime)
    network:backward({D.s,D.a},grad)
    t_err = torch.pow(o-D.sPrime,2):sum(2)
    local o_err = err_network:forward{D.s,D.a,o}
    loss = loss + mse_crit:forward(o_err,t_err)
    local err_grad =  mse_crit:backward(o_err,t_err)
    err_network:backward({D.s,D.a,o},err_grad)
    return loss,dw
end

refresh = 1e2
cumloss = 0
for i = 1,1e4 do
    _,batchloss = optim.adam(train,w,config)
    cumloss = cumloss + batchloss[1]
    if i % refresh == 0 then
        print(i,cumloss/refresh,w:norm(),dw:norm())
        gnuplot.figure(1)
        gnuplot.imagesc(network.output)
        gnuplot.figure(2)
        gnuplot.bar(err_network.output)
        gnuplot.figure(3)
        gnuplot.bar(t_err)
        --gnuplot.imagesc(D.sPrime)
        sys.sleep(1)
        
        cumloss = 0
    end
end
