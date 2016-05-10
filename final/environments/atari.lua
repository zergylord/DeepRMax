env = {}
local alewrap = require 'alewrap'
local validA,num_steps,refresh
require '../Scale'
local prep = nn.Scale(84,84)
--[[
--param table options:
--game: which rom (default 'pong')
--actrep: num of extra frames between observations (default 3)
--num_hist: num of previous frames per observation (default 3)
--]]
function env.setup(params)
    params = params or {}
    params.game_path = '../../roms/'
    params.env = params.env or 'pong'
    params.actrep = params.actrep or 3
    num_steps = params.num_steps or 1e6
    refresh = params.refresh or 1e3
    num_hist = params.num_hist or 3
    game = alewrap.GameEnvironment(params)
    validA = game:getActions()
    act_dim = #validA

    state_dim = 84*84
    A = torch.eye(act_dim)

    --init viz variables----------------------------------
    bonus_hist = torch.zeros(num_steps/refresh)
end
--[[
--clears internal state and returns initial state information
--]]
function env.reset()
    local screen = game:nextRandomGame() --newGame()
    return prep:forward(screen)
end
--[[
--returns observation and exact state
--]]
function env.step(s,a)
    local nextScreen,r,term = game:step(validA[a])
    return r,prep:forward(nextScreen)
end
--[[
--returns one-hot vector for action a
--]]
function env.get_action(a)
    return A[a]
end
--[[ return all possible states
--]]
function env.get_all_states()
    return S
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
function env.get_info(network,err_network,pred_network,q_network) 
end
