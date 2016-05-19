env = {}
local S,A,T,correct,all_state,all_action,all_statePrime,num_steps,refresh
local bonus_hist,bonus,replay_visits,visits
local visits_over_time,visits_over_time_norm,visits_over_time_per_state,visits_over_time_per_state_norm
--[[
--param table options:
--actions: number of actions (default 4)
--states: number of states (default 30)
--]]
function env.setup(params)
    params = params or {}
    env.act_dim = params.actions or 4
    num_state = params.states or 30
    env.in_dim = num_state
    env.state_dim = env.in_dim
    num_steps = params.num_steps or 1e6
    env.refresh = params.refresh or 1e3
    A = torch.eye(env.act_dim)
    T = torch.ones(num_state,env.act_dim)
    correct = torch.ones(num_state)
    for i = 1,num_state-1 do
        local action = torch.random(env.act_dim)
        T[i][action] = i+1
        correct[i] = action
    end
    --setup MDP
    S = torch.eye(num_state)
    --S = torch.tril(torch.ones(num_state,num_state))

    all_statePrime = torch.zeros(num_state*env.act_dim,num_state)
    all_state = torch.zeros(num_state*env.act_dim,num_state)
    all_action = torch.zeros(num_state*env.act_dim,env.act_dim)
    for s=1,num_state do
        for a=1,env.act_dim do
            all_state[env.act_dim*(s-1)+a] = S[s]
            all_action[env.act_dim*(s-1)+a] = A[a]
            all_statePrime[env.act_dim*(s-1)+a] = S[T[s][a]]
        end
    end
    env.s = 1
    --init viz variables----------------------------------
    bonus_hist = torch.zeros(num_steps/env.refresh)
    bonus = torch.zeros(num_state,env.act_dim)
    replay_visits = torch.zeros(num_state,env.act_dim)
    visits = torch.zeros(num_state,env.act_dim)
    visits_over_time = torch.zeros(num_steps/env.refresh,num_state)
    bonus_over_time = torch.zeros(num_steps/env.refresh,num_state*env.act_dim)
    bonus_over_time_norm = torch.zeros(num_steps/env.refresh,num_state*env.act_dim)
    bonus_over_time_per_state = torch.zeros(num_steps/env.refresh,num_state)
    bonus_over_time_per_state_norm = torch.zeros(num_steps/env.refresh,num_state)
end
--[[
--clears internal state and returns initial state information
--]]
function env.reset()
    env.s = 1
    return S[1],1
end
--[[
--called every step to record relevant statistics (e.g. visitation counts)
--]]
function env.update_step_stats(s,a)
    visits[s][a] = visits[s][a] + 1
end
--[[
--returns observation and exact state
--]]
function env.step(a)
    env.update_step_stats(env.s,a)
    local sPrime = T[env.s][a]
    env.s = sPrime
    local r  = 0
    if sPrime == num_state then
        r = 1
    end
    return r,S[sPrime],sPrime==num_state
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
--called for every s,a pair considered during replay
--to record relevant statistics (e.g. bonus percentage(
--]]
function env.update_replay_stats(obs,a,chance_unknown)
    local s = obs:float():nonzero()[1][1]
    replay_visits[s ][a] = replay_visits[s ][a] + 1
    bonus[s ][a] = bonus[s][a] + chance_unknown
end
--[[
--called every 'refresh' steps, normally to plot data
--]]
function env.get_info(t,reward_hist,network,err_network,pred_network,q_network) 
    gnuplot.figure(1)
    gnuplot.raw("set multiplot layout 2,5 columnsfirst")


    if q_network then
        local Q
        Q = q_network:forward(env.get_all_states():cuda())
        gnuplot.raw("set title 'Q-values' ")
        gnuplot.raw('set xrange [' .. .5 .. ':' .. num_state+.5 .. '] noreverse')
        gnuplot.raw('set yrange [*:*] noreverse')
        gnuplot.plot({Q:mean(2)},{Q:max(2):float()},{Q:min(2):float()})
    else
        gnuplot.raw("set title 'Q-values' ")
        gnuplot.raw('set xrange [' .. .5 .. ':' .. num_state+.5 .. '] noreverse')
        gnuplot.raw('set yrange [0:1] noreverse')
        gnuplot.plot({Q:mean(2)},{Q:max(2):float()},{Q:min(2):float()})
    end


    
    gnuplot.raw("set title '% R-Max bonus over time' ")
    gnuplot.raw('set xrange [' .. 0 .. ':' .. t/env.refresh+1 .. '] noreverse')
    gnuplot.raw('set yrange [0:.1] noreverse')
    local avg_bonus = bonus:cdiv(replay_visits+1)
    print(avg_bonus:mean())
    bonus_hist[t/env.refresh] = avg_bonus:mean()
    gnuplot.plot(bonus_hist[{{1,t/env.refresh}}])
    
    
    
    bonus_over_time[t/env.refresh] = avg_bonus
    bonus_over_time_per_state[t/env.refresh] = avg_bonus:mean(2)
    bonus_over_time_norm[t/env.refresh] = avg_bonus:div(avg_bonus:max())
    bonus_over_time_per_state_norm[t/env.refresh] = avg_bonus:mean(2):div(avg_bonus:mean(2):max())
    gnuplot.raw("set title 'bonus % over time' ")
    gnuplot.imagesc(bonus_over_time[{{1,t/env.refresh}}])
    gnuplot.raw("set title 'bonus % over time per state' ")
    gnuplot.imagesc(bonus_over_time_per_state[{{1,t/env.refresh}}])
    gnuplot.raw("set title 'bonus % over time normed' ")
    gnuplot.imagesc(bonus_over_time_norm[{{1,t/env.refresh}}])
    gnuplot.raw("set title 'bonus % over time per state normed' ")
    gnuplot.imagesc(bonus_over_time_per_state_norm[{{1,t/env.refresh}}])

    print(visits:sum(2)[{{},1}])
    visits_over_time[t/env.refresh] = visits:sum(2)[{{},1}]

    cur_known = torch.zeros(num_state,env.act_dim)
    cur_pred = torch.zeros(num_state*env.act_dim,env.in_dim)
    cur_actual_pred = torch.zeros(num_state*env.act_dim,env.in_dim)
    cur_err = torch.zeros(num_state,env.act_dim)
    cur_actual_err = torch.zeros(num_state,env.act_dim)

    network:forward{all_state:cuda(),all_action:cuda()}
    cur_pred = pred_network.output:float()
    cur_actual_pred = all_statePrime
    cur_err = get_knownness(err_network.output):reshape(num_state,env.act_dim)
    cur_actual_err = BCE(cur_pred,cur_actual_pred):reshape(num_state,env.act_dim)
    
    gnuplot.raw("set title 'current pred' ")
    gnuplot.imagesc(cur_pred)
    gnuplot.raw("set title 'pred diff' ")
    gnuplot.imagesc(cur_actual_pred - cur_pred)
    gnuplot.raw("set title 'current pred pred err' ")
    gnuplot.imagesc(cur_err)
    gnuplot.raw("set title 'current actual pred err' ")
    gnuplot.imagesc(cur_actual_err)

    


    gnuplot.raw('unset multiplot')
    visits:zero()
    replay_visits:zero()
    bonus:zero()
end
