require 'ReplayTable'
require 'cunn'
state_dim = 100
opt = {}
opt.gpu = true
D = ReplayTable.init(state_dim,1,false)
data = torch.eye(100)
for i=1,100 do
    D:add(data[i],1,1,false)
end
s = D:get_samples(1e4)
print(s:sum(1))

