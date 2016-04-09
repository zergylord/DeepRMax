require 'gnuplot'
require 'nngraph'
require 'optim'
require 'distributions'
f = function(x,c) return torch.cdiv(x,x+c) end
fPrime = function(x,c) return -torch.cdiv(x,torch.pow(x+c,2)) + (x+c):pow(-1) end
x = torch.linspace(0,10)
gnuplot.figure(1)
gnuplot.axis{'','',0,1}
--[[
for i=1,10 do
c = .25*2^(i-1)
gnuplot.plot({x,f(x,c)},{x,fPrime(x,c)})
sys.sleep(1)
end
--]]

in_dim = 4
hid_dim = 100
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
prob = nn.SoftPlus()(nn.Linear(hid_dim,1)(hid))
denom = nn.AddConstant(40)(prob)
output = nn.CDivTable(){prob,denom}
network = nn.gModule({input},{output,prob})
w,dw = network:getParameters()
network:zeroGradParameters()

mb_dim = 32
d_dist = torch.Tensor{.55,.1,.1,.25}
g_dist = torch.Tensor{.25,.25,.25,.25}

A = torch.eye(in_dim)

get_data = function(data,dim)
    data[{{1,dim/2}}] = distributions.cat.rnd(dim/2,d_dist,{categories=A})
    data[{{dim/2+1,-1}}] = distributions.cat.rnd(dim/2,g_dist,{categories=A})
end
bce_crit = nn.BCECriterion()
data = torch.zeros(mb_dim,in_dim)
target = torch.zeros(mb_dim,1)
target[{{1,mb_dim/2}}] = 1
train = function(w)
    get_data(data,mb_dim)
    local output = network:forward(data)[1]
    local loss = bce_crit:forward(output,target)
    local grad = bce_crit:backward(output,target)
    network:backward(data,{grad,torch.zeros(mb_dim,1)})

    return loss,dw
end
config = {learningRate = 1e-4}
cumloss = 0
num_steps = 5e4
refresh = 1e3
for i=1,num_steps do
    _,batchloss = optim.adam(train,w,config)
    cumloss = cumloss + batchloss[1]
    if i % refresh == 0 then
        print(d_dist)
        print(i,cumloss,w:norm(),dw:norm())
        network:forward(A)
        gnuplot.figure(1)
        print(network.output[2]:sum())
        gnuplot.bar(network.output[1][{{},1}])
        gnuplot.axis{.5,in_dim+.5,0,1}
        gnuplot.figure(2)
        --gnuplot.bar(network.output[2][{{},1}])
        --gnuplot.axis{.5,in_dim+.5,0,30}
        gnuplot.bar(network.output[2][{{},1}]:div(network.output[2]:sum()))
        --gnuplot.bar(torch.cdiv(d_dist,d_dist+g_dist))
        gnuplot.axis{.5,in_dim+.5,0,1}
        cumloss = 0
    end
end
