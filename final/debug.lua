require 'nngraph'
require 'cunn'
env = {}
env.num_hist = 3
env.image_size = {84,84}
env.act_dim = 4
input = nn.Identity()()
local view = nn.View(-1,env.num_hist+1,env.image_size[1],env.image_size[2])(input)
local conv1 = nn.ReLU()(nn.SpatialConvolution(env.num_hist+1,32,8,8,4,4)(view))
local conv2 = nn.ReLU()(nn.SpatialConvolution(32,64,4,4,2,2)(conv1))
local conv3 = nn.ReLU()(nn.SpatialConvolution(64,64,3,3,1,1)(conv2))
local conv_dim = 64*7*7
local last_hid = nn.ReLU()(nn.Linear(conv_dim,512)(nn.View(-1,conv_dim)(conv3)))
output = nn.Linear(512,env.act_dim)(last_hid)
q_network = nn.gModule({input},{output})
print(q_network:forward(torch.rand(4*84*84)))
print(q_network:forward(torch.rand(10*4*84*84)))
