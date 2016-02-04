require 'nngraph'
in_dim = 100
hid_dim = 100
out_dim = 100
input = nn.Identity()()
hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
out = nn.Linear(hid_dim,out_dim)(hid)
network = nn.gModule({input},{out})
timer = torch.Timer()
refresh = 1e5
cum = 0
for i=1,1e10 do
	network:forward(torch.rand(in_dim))
	network:backward(torch.rand(in_dim),torch.rand(out_dim))
	if i % refresh == 0 then
		print(timer:time().real)
		timer:reset()
	end
end

