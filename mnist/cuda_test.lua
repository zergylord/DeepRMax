require 'cunn'
require 'nngraph'
--torch.setnumthreads(1)
timer = torch.Timer()
dim = 10000
batch = 1000
iter = 100
--
input = nn.Identity()()
hid1 = nn.ReLU()(nn.Linear(dim,dim)(input))
hid2 = nn.ReLU()(nn.Linear(dim,dim)(hid1))
last_hid = nn.ReLU()(nn.Linear(dim,dim)(hid2))
network = nn.gModule({input},{last_hid})
for i = 1,iter do
network:forward(torch.rand(batch,dim))
end
--]]
--[[
input = nn.Identity():cuda()()
hid1 = nn.ReLU():cuda()(nn.Linear(dim,dim):cuda()(input))
hid2 = nn.ReLU():cuda()(nn.Linear(dim,dim):cuda()(hid1))
last_hid = nn.ReLU():cuda()(nn.Linear(dim,dim):cuda()(hid2))
network = nn.gModule({input},{last_hid})
for i=1,iter do
network:forward(torch.rand(batch,dim):cuda())
end
--]]
print(timer:time().real)
