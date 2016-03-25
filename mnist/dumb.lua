require 'distributions'
require 'nngraph'
require 'optim'
require 'gnuplot'
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(1,10)(input))
output = nn.Sigmoid()(nn.Linear(10,1)(hid))
network = nn.gModule({input},{output})
w,dw = network:getParameters()
mb_dim = 32
mb_half = mb_dim/2
config = {learningrate = 1e-3}
--target = torch.ones(mb_dim,1)
--target[{{1,mb_half}}] = 0
crit = nn.BCECriterion()
mu = 5
function train(x)
    if x ~= w then
        w:copy(x)
    end
    network:zeroGradParameters()
    local data = torch.randn(mb_dim,1):add(mu)
    local target = torch.rand(mb_dim,1):lt(.5):double()
    --local data = distributions.norm.qtl(torch.linspace(.0001,.9999,mb_dim),0,1):reshape(mb_dim,1)
    output = network:forward(data)
    loss = crit:forward(output,target)
    grad = crit:backward(output,target)
    network:backward(data,grad)
    return loss,dw
end
num_steps = 1e5
refresh = 1e2
loss_over_time = torch.zeros(num_steps)
var_over_time = torch.zeros(num_steps)
plots = {}
--data = distributions.norm.qtl(torch.linspace(.00001,.99999),0,1):reshape(100,1)
data = torch.linspace(-4,4):reshape(100,1):add(mu)
for i = 1,num_steps do
    _,batchloss = optim.adam(train,w,config)
    var_over_time[i] = output:var()
    loss_over_time[i] = batchloss[1]
    if i % refresh == 0 then
        --gnuplot.plot({var_over_time},{loss_over_time})
        table.insert(plots,{data[{{},1}],network:forward(data):clone()})
        gnuplot.plot(plots)
        --gnuplot.axis{data:min(),data:max(),.45,.55}
        --sys.sleep(.1)
    end
end
