require 'nngraph'
require 'optim'
require 'gnuplot'
require 'ReplayTable'
--require 'hdf5'
require 'cunn'
require 'util/BCE'
--torch.manualSeed(123)
--cutorch.manualSeed(123)
--torch.setnumthreads(1)
local timer = torch.Timer()
torch.setdefaulttensortype('torch.FloatTensor')
gnuplot.figure(1)

cmd = torch.CmdLine()
--set hyper parameters-------------
cmd:option('-update_freq',4,'steps per weight update step')
cmd:option('-use_rmax',true,'use r-max style exploration')
cmd:option('-mb_dim',32,'size of minibatch')
cmd:option('-rmax',.1,'maximum reward value for purposes of exploration')
cmd:option('-refresh',1e4,'steps until information is displayed')
cmd:option('-num_steps',1e7,'total steps to run')
cmd:option('-use_egreedy',true,'use epsilon greedy action selection')
cmd:option('-use_target_network',true,'use a target network for updates')
cmd:option('-target_refresh',3e4,'how often to copy network into target network')
cmd:option('-learn_start',5e4,'when to start making weight updates')
cmd:option('-clip_delta',true,'false or constant value for gradients')
cmd:option('-gamma',.99,'discount factor')
cmd:option('-q_learning_rate',2.5e-4,'learning rate for q network')
cmd:option('-environment','atari','training task to use')
cmd:option('-num_frames',4,'number of consequtive frames for input')
opt = cmd:parse(arg)
print(opt)
q_config = {
    learningRate  = opt.q_learning_rate
    }
if opt.use_egreedy then
epsilon = torch.ones(opt.learn_start):cat((-torch.linspace(0,.9,1e6)+1):cat(torch.ones(opt.num_steps-1e6-opt.learn_start):mul(.1)))
end
net_reward = 0
reward_hist = torch.zeros(opt.num_steps/opt.refresh)


--select environment---------------------------------------------
require('environments/' .. opt.environment)
env.setup{refresh=opt.refresh,num_steps=opt.num_steps}
D = ReplayTable.init(env.state_dim,opt.num_frames,env.byte_storage)

--select exploration method------------------------------------
require 'models/PPE'
--require 'train_pred_GAN.lua'
setup(env)


--setup value function------------------------------------
local input,output
if env.spatial then
    print(' using conv net')
    input = nn.Identity()()
    local view = nn.View(-1,D.num_frames,env.image_size[1],env.image_size[2])(input)
    local conv1 = nn.ReLU()(nn.SpatialConvolution(D.num_frames,32,8,8,4,4)(view))
    local conv2 = nn.ReLU()(nn.SpatialConvolution(32,64,4,4,2,2)(conv1))
    local conv3 = nn.ReLU()(nn.SpatialConvolution(64,64,3,3,1,1)(conv2))
    local conv_dim = 64*7*7
    local last_hid = nn.ReLU()(nn.Linear(conv_dim,512)(nn.View(-1,conv_dim)(conv3)))
    lin = nn.Linear(512,env.act_dim,false)(last_hid)
    output = nn.Add(1,true)(lin)
else
    print('using mlp')
    local hid_dim = 100
    input = nn.Identity()()
    local hid = nn.ReLU()(nn.Linear(in_dim,hid_dim)(input))
    output =nn.Linear(hid_dim,env.act_dim)(hid)
end
q_network = nn.gModule({input},{output})
q_network = q_network:cuda()
q_w,q_dw = q_network:getParameters()
if use_target_network then
    target_network = q_network:clone()
end
mse_crit = nn.MSECriterion():cuda()






--setup experience replay------------------------------
blank = torch.zeros(opt.mb_dim,env.act_dim)
local get_data = function(data,action_data,dataPrime)
    data[{{}}] = mb_s
    blank:zero()
    action_data[{{}}] = blank:scatter(2,mb_a:long():view(opt.mb_dim,1),1)
    if D.num_frames > 1 then
        dataPrime[{{}}] = D.get_pred_state(mb_sPrime)
    else
        dataPrime[{{}}] = mb_sPrime
    end
end
set_data_func(get_data)
final_time = -1
possible = {}
possible[1] = torch.zeros(opt.mb_dim*env.act_dim,in_dim):cuda()
possible[2] = torch.zeros(opt.mb_dim*env.act_dim,env.act_dim):cuda()
for i =1,opt.mb_dim do
    for a = 1,env.act_dim  do
        possible[2][opt.mb_dim*(a-1)+i][a] = 1
    end
end
target = torch.zeros(opt.mb_dim,env.act_dim):cuda()
target_mask = torch.zeros(opt.mb_dim,env.act_dim,1):byte()
act_grad = torch.zeros(env.act_dim*opt.mb_dim,env.act_dim):cuda()
aind = torch.LongTensor(opt.mb_dim*env.act_dim,1):cuda()
for a=1,env.act_dim do
    for i=1,opt.mb_dim do
        aind[opt.mb_dim*(a-1)+i] = a
    end
end
local function q_train(x)
    q_network:zeroGradParameters()
    local o = q_network:forward(mb_s)
    --
    local loss = mse_crit:forward(o,target)
    act_grad = mse_crit:backward(o,target)
    --]]
    --[[
    act_grad = -(target-o)
    local loss = 0
    --]]
    act_grad[target_mask:eq(0)] = 0
    if opt.clip_delta then
        act_grad[act_grad:gt(1)] = 1
        act_grad[act_grad:lt(-1)] = -1
    end
    q_network:backward(mb_s,act_grad)
    return loss,q_dw
end
--main loop------------------------------------------------------
cumloss =0 
s = env.reset()
for t=1,opt.num_steps do
    r = 0
    --select action
    if (opt.use_egreedy and torch.rand(1)[1] < epsilon[t]) or (t <= opt.learn_start) then
        a = torch.random(env.act_dim)
    else
        if D.num_frames > 1 then
            full_s = D:get_past(s):cuda()
        else
            full_s = s:cuda()
        end
        local vals = q_network:forward(full_s)
        _,a = vals:max(vals:dim())
        a = a:squeeze()
    end

    --perform action
    r,sPrime,term = env.step(a)
    if r > 0 then
        r = 1
    elseif r< 0 then
        r = -1
    end
    net_reward = net_reward + r


    --record history
    D:add(s,a,r,term)

    --update model params
    if t > opt.learn_start and t % opt.update_freq == 0 then
        --gotta re-perm if runnning train_dis multiple times
        mb_s,mb_a,mb_r,mb_sPrime,mb_term = D:get_samples(opt.mb_dim)
        if opt.use_rmax then
            --update adver nets
            _,batchloss = optim.adam(train,w,config)
            cumloss = cumloss + batchloss[1]
        end
        --update Q
        if opt.use_target_network then
            _,qind = q_network:forward(mb_sPrime):max(2)
            qPrime = target_network:forward(mb_sPrime):gather(2,qind)
            --qPrime,qind = target_network:forward(mb_sPrime):max(2)
        else
            qPrime,qind = q_network:forward(mb_sPrime):max(2)
        end

        if opt.use_rmax then
            for i=1,opt.mb_dim do
                for a=1,env.act_dim do
                    possible[1][opt.mb_dim*(a-1)+i] = mb_s[i]
                end
            end
            C = network:forward(possible)
        end
    
        
        target:zero()
        target_mask:zero()
        for i=1,opt.mb_dim do
            --you can experience all actions under threshold, since they all go to heaven!
            local known_flag = true
            if opt.use_rmax then
                for a = 1,env.act_dim do
                    local ind = opt.mb_dim*(a-1)+i
                    local unknown, chance_unknown = get_knownness(C,ind)
                    env.update_replay_stats(mb_s[i],a,chance_unknown)

                    if  unknown then
                        if a == mb_a[i] then
                            known_flag = false
                        end
                        target[i][a] = opt.rmax
                        target_mask[i][a] = 1
                    end
                end
            end
            --only experienced actions can be updated over threshold
            if known_flag then
                --print('known!')
                target_mask[i][mb_a[i] ] = 1
                if mb_term[i] == 1 then
                    target[i][mb_a[i] ] = mb_r[i]
                else
                    target[i][mb_a[i] ] = mb_r[i]+opt.gamma*torch.max(qPrime[i])
                end
            end
        end
        --update value function----------------------------
        _,batchloss = optim.adam(q_train,q_w,q_config)
        if use_target_network and t % opt.target_refresh == 0 then
            target_network = q_network:clone()
        end
    end
    if term then
        s = env.reset()
    else
        s = sPrime 
    end
    if t % 1000 == 0 then collectgarbage() end
    if t % opt.refresh == 0 then
        reward_hist[t/opt.refresh] = net_reward/opt.refresh
        env.get_info(t,reward_hist,network,err_network,pred_network,q_network)
        --[[
        local frames = mb_s[1]:view(4,84*84)
        for f=1,4 do
            gnuplot.imagesc(frames[f]:view(84,84))
            sys.sleep(1)
        end
        --]]


        torch.save('w.t7',{q_w,w})

        print(t,net_reward/opt.refresh,cumloss,q_w:norm(),q_dw:norm(),timer:time().real)
        timer:reset()
        cumloss = 0
        if num_state and net_reward/opt.refresh > ((1/num_state)*(.8)) then
            final_time = t
            print(final_time)
            break
        end
        net_reward = 0
    end

end




