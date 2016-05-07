env = {}
local S,A,T,correct,all_state,all_action,all_statePrime
--[[
--param table options:
--size: tiles per side of grid (default 10)
--]]
function env.setup(params)
    params = params or {}
    num_act = 4
    side = params.size or 10
    num_state = side^2
    A = torch.eye(num_act)
    T = torch.ones(num_state,num_act)
    for i = 1,side do
        for j = 1,side do
            --up
            local row = ((i-1-1) % side) + 1
            local col = j
            T[side*(i-1)+j][1] = side*(row-1)+col
            --left
            local row = i
            local col = ((j-1-1) % side) + 1
            T[side*(i-1)+j][2] = side*(row-1)+col
            --down
            local row = ((i-1+1) % side) + 1
            local col = j
            T[side*(i-1)+j][3] = side*(row-1)+col
            --right
            local row = i
            local col = ((j-1+1) % side) + 1
            T[side*(i-1)+j][4] = side*(row-1)+col
        end
    end
    --setup MDP
    S = torch.eye(num_state)
    --S = torch.tril(torch.ones(num_state,num_state))

    all_statePrime = torch.zeros(num_state*num_act,num_state)
    all_state = torch.zeros(num_state*num_act,num_state)
    all_action = torch.zeros(num_state*num_act,num_act)
    for s=1,num_state do
        for a=1,num_act do
            all_state[num_act*(s-1)+a] = S[s]
            all_action[num_act*(s-1)+a] = A[a]
            all_statePrime[num_act*(s-1)+a] = S[T[s][a]]
        end
    end
end
--[[
--clears internal state and returns initial state information
--]]
function env.reset()
    local s = torch.random(num_state)
    return S[s],s
end
--[[
--returns observation and exact state
--]]
function env.step(s,a)
    sPrime = T[s][a]
    return 0,S[sPrime],sPrime
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



