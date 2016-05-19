env = {}
local alewrap = require 'alewrap'
local validA,num_steps,refresh
require '../Scale'
local prep = nn.Scale(84,84)
local H,last_obs
--[[
--param table options:
--game: which rom (default 'pong')
--actrep: num of extra frames between observations (default 3)
--num_hist: num of previous frames per observation (default 3)
--]]
function env.setup(params)
    params = params or {}
    params.game_path = '../roms/'
    params.env = params.env or 'pong'
    params.actrep = params.actrep or 3
    num_steps = params.num_steps or 1e6
    refresh = params.refresh or 1e3
    game = alewrap.GameEnvironment(params)
    validA = game:getActions()


    --init visible members
    env.state_dim = 84*84
    env.act_dim = #validA
    env.num_hist = params.num_hist or 3
    env.image_size = torch.Tensor{84,84}
    env.in_dim = env.state_dim*(env.num_hist+1)
    env.byte_storage = true
    env.spatial = true

    H = torch.zeros(env.num_hist,84,84):float()
    A = torch.eye(env.act_dim)
    --init viz variables----------------------------------
    bonus_hist = torch.zeros(num_steps/refresh)
end
--[[
--clears internal state and returns initial state information
--]]
function env.reset()
    local screen = game:nextRandomGame() --newGame()
    if env.num_hist > 0 then
        H:zero()
        last_obs = prep:forward(screen[1]):cat(H,1)
    else
        last_obs = prep:forward(screen[1])
    end
    return last_obs:view(env.in_dim)
end
--[[
--returns observation and exact state
--]]
function env.step(a)
    local nextScreen,r,term = game:step(validA[a])
    if env.num_hist > 0 then
        if env.num_hist > 1 then
            H[{{2,env.num_hist}}] = H[{{1,env.num_hist-1}}]:clone()
        end
        H[1] = last_obs[1]
        last_obs = prep:forward(nextScreen[1]):cat(H,1)
    else
        last_obs = prep:forward(nextScreen[1])
    end
    return r,last_obs:view(env.in_dim),term
end
--[[
--returns one-hot vector for action a
--]]
function env.get_action(a)
    return A[a]
end
--[[
--get the predictable part of the state
--]]
function env.get_pred_state(s)
    if s:dim() == 1 then
        return s[{{1,env.state_dim}}]
    else
        return s[{{},{1,env.state_dim}}]
    end
end

--[[ return all possible states
--]]
function env.get_all_states()
    return nil
end
--[[ return all possible s,a,sPrime combinations
--]]
function env.get_complete_dataset()
    return nil
end

--[[
--called every step to record relevant statistics (e.g. visitation counts)
--]]
function env.update_step_stats(s,a)
end
--[[
--called for every s,a pair considered during replay
--to record relevant statistics (e.g. bonus percentage(
--]]
function env.update_replay_stats(s,a,chance_unknown)
end
--[[
--called every 'refresh' steps, normally to plot data
--]]
function env.get_info(t,reward_hist,network,err_network,pred_network,q_network) 
    gnuplot.plot(reward_hist[{{1,t/refresh}}])
end
--[[
env.setup()
s = env.reset()
sPrime = env.step(s,1)
--]]
