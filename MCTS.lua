num_states = 10
num_actions = 4
num_steps = 1e5
max_depth = 12
T = torch.ones(num_states,num_actions)
for s=1,num_states-1 do
    T[s][torch.random(num_actions)] = s+1
end
print(T)
for t=1,num_steps do
    real_s = 1
    s = real_s
    Q = torch.rand(num_states,num_actions)
    for k=1,num_rollouts do
        for d = 1,max_depth do
            _,a = Q[s]:max(1)
            s = T[s][a]




