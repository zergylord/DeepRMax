require 'ReplayTable'
require 'cunn'
state_dim = 100
D = ReplayTable.init(state_dim,4,true)
for i=1,100 do
    D:add(torch.rand(state_dim),1,1,false)
end
frame = torch.rand(state_dim)
for i=1,2e4 do
    if i % 1e3 == 0 then
        print(i)
    end
    D:add(frame,1,1,false)
    D:get_samples(32)
end

