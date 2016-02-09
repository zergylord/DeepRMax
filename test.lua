require 'nngraph'
require 'optim'
mb_dim = 32
in_dim = 100
hid_dim = 100
out_dim = 100
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
out = nn.Linear(hid_dim,out_dim)(hid)
network = nn.gModule({input},{out})
w,dw = network:getParameters()
timer = torch.Timer()
refresh = 1e3
cum = 0
train = function(x)
    if x ~= w then
        w:copy(x)
    end
    network:forward(torch.rand(mb_dim,in_dim))
    network:backward(torch.rand(mb_dim,in_dim),torch.rand(mb_dim,out_dim))
    return 0,dw
end
config = {learningRate = 1e-3}
for i=1,1e10 do
    optim.rmsprop(train,w,config)
    if i % refresh == 0 then
        print(timer:time().real)
        timer:reset()
        collectgarbage()
    end
end

