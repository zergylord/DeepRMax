env = {}
local alewrap = require 'alewrap'
local game 
--[[
--param table options:
--game: which rom (default 'pong')
--]]
function env.setup(params)
    params = params or {}
    rom = params.game or 'pong'
    game = alewrap.createEnv('../../roms/' .. rom .. '.bin', {})
    local validA = game:actions()

    num_state = params.states or 30
    A = torch.eye(act_dim)
    T = torch.ones(num_state,act_dim)
    correct = torch.ones(num_state)
    for i = 1,num_state-1 do
        local action = torch.random(act_dim)
        T[i][action] = i+1
        correct[i] = action
    end
    --setup MDP
    S = torch.eye(num_state)
    --S = torch.tril(torch.ones(num_state,num_state))

    all_statePrime = torch.zeros(num_state*act_dim,num_state)
    all_state = torch.zeros(num_state*act_dim,num_state)
    all_action = torch.zeros(num_state*act_dim,act_dim)
    for s=1,num_state do
        for a=1,act_dim do
            all_state[act_dim*(s-1)+a] = S[s]
            all_action[act_dim*(s-1)+a] = A[a]
            all_statePrime[act_dim*(s-1)+a] = S[T[s][a]]
        end
    end
    --init viz variables----------------------------------
    bonus_hist = torch.zeros(num_steps/refresh)
    bonus = torch.zeros(num_state,act_dim)
    replay_visits = torch.zeros(num_state,act_dim)
    visits = torch.zeros(num_state,act_dim)
    visits_over_time = torch.zeros(num_steps/refresh,num_state)
    bonus_over_time = torch.zeros(num_steps/refresh,num_state*act_dim)
    bonus_over_time_norm = torch.zeros(num_steps/refresh,num_state*act_dim)
    bonus_over_time_per_state = torch.zeros(num_steps/refresh,num_state)
    bonus_over_time_per_state_norm = torch.zeros(num_steps/refresh,num_state)
end
--[[
--clears internal state and returns initial state information
--]]
function env.reset()
    return S[1],1
end
--[[
--returns observation and exact state
--]]
function env.step(s,a)
    local sPrime = T[s][a]
    local r  = 0
    if sPrime == num_state then
        r = 1
    end
    return r,S[sPrime],sPrime
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
    return all_state,all_action,all_statePrime
end

--[[
--called every step to record relevant statistics (e.g. visitation counts)
--]]
function env.update_step_stats(s,a)
    visits[s][a] = visits[s][a] + 1
end
--[[
--called for every s,a pair considered during replay
--to record relevant statistics (e.g. bonus percentage(
--]]
function env.update_replay_stats(s,a,chance_unknown)
    replay_visits[s[i] ][a] = replay_visits[s[i] ][a] + 1
    bonus[s[i] ][a] = bonus[s[i] ][a] + chance_unknown
end
--[[
--called every 'refresh' steps, normally to plot data
--]]
function env.get_info(network,err_network,pred_network,q_network) 
end
