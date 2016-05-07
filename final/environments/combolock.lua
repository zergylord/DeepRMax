env = {}
local S,A,T,correct,all_state,all_action,all_statePrime
--[[
--param table options:
--actions: number of actions (default 4)
--states: number of states (default 30)
--]]
function env.setup(params)
    params = params or {}
    act_dim = params.actions or 4
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



