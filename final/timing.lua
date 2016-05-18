require 'nngraph'
require 'cunn'
require 'optim'
time = sys.clock()
env = {}
env.image_size = {84,84}
env.act_dim = 4
env.num_hist = 3
input = nn.Identity()()
local view = nn.View(-1,env.num_hist+1,env.image_size[1],env.image_size[2])(input)
local conv1 = nn.ReLU()(nn.SpatialConvolution(env.num_hist+1,32,8,8,4,4)(view))
local conv2 = nn.ReLU()(nn.SpatialConvolution(32,64,4,4,2,2)(conv1))
local conv3 = nn.ReLU()(nn.SpatialConvolution(64,64,3,3,1,1)(conv2))
local conv_dim = 64*7*7
local last_hid = nn.ReLU()(nn.Linear(conv_dim,512)(nn.View(-1,conv_dim)(conv3)))
output = nn.Linear(512,env.act_dim)(last_hid)
q_network = nn.gModule({input},{output})
q_network = q_network:cuda()
q_w,q_dw = q_network:getParameters()

data = torch.rand(32,4,84,84):cuda()
target = torch.rand(32,4):cuda()
crit = nn.MSECriterion():cuda()
function train(w)
    q_network:zeroGradParameters()
    o = q_network:forward(data)
    loss = crit:forward(o,target)
    grad = crit:backward(o,target)
    q_network:backward(data,grad)
    return loss,q_dw
end
config = {learningRate=1e-3}
for i=1,1e5 do
    optim.adam(train,q_w,config)
    if i % 1e3 == 0 then
        print(i,sys.clock()-time)
        time = sys.clock()
    end
end

    

