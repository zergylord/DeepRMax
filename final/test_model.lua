require 'nngraph'
require 'cunn'
require 'gnuplot'
input = nn.Identity()()
 view = nn.View(-1,4,84,84)(input)
 just_conv = nn.SpatialConvolution(4,32,8,8,4,4)
 conv1 = nn.ReLU()(just_conv(view))
 conv2 = nn.ReLU()(nn.SpatialConvolution(32,64,4,4,2,2)(conv1))
 conv3 = nn.ReLU()(nn.SpatialConvolution(64,64,3,3,1,1)(conv2))
 conv_dim = 64*7*7
 last_hid = nn.ReLU()(nn.Linear(conv_dim,512)(nn.View(-1,conv_dim)(conv3)))
output = nn.Linear(512,3)(last_hid)
q_network = nn.gModule({input},{output})
q_network = q_network:cuda()
q_w,q_dw = q_network:getParameters()
vid = unpack(torch.load('frames.dat'))
require 'image'
q_network:forward(image.scale(vid[{{1,4}}],84,84,'bilinear'):float():div(255):cuda())
gnuplot.imagesc(just_conv.output[1][1])
sys.sleep(1)
old = q_w:clone()
q_w = unpack(torch.load('w.t7'))
q_network:forward(image.scale(vid[{{1,4}}],84,84,'bilinear'):float():div(255):cuda())
gnuplot.imagesc(just_conv.output[1][1])
