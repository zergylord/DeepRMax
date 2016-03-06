require 'gnuplot'
num_state = 100
Q = torch.zeros(num_state,4)
T = torch.ones(num_state,4)
correct = torch.ones(num_state)
for i = 1,num_state-1 do
    action = torch.random(4)
    T[i][action] = i+1
    correct[i] = action
end





s = 1
epsilon = .1
alpha = .1
gamma = .9
net_reward = 0
refresh = 1e3
rmax = true
C = torch.zeros(num_state,4)
thresh = 1
for t=1,1e6 do
    r = 0
    --select action
    if rmax then
        _,a = torch.max(Q[s],1)
        a = a[1]
        for i = 1,4 do
            if C[s][i] < thresh then
               a = i
               r = (1-gamma)*1
            end
        end
    elseif torch.rand(1)[1] < epsilon then
        a = torch.random(4)
    else
        _,a = torch.max(Q[s],1)
        a = a[1]
        --a = correct[s]
    end

    --perform action
    sPrime = T[s][a]
    C[s][a] = C[s][a] + 1
    if sPrime == num_state then
        r = 1
        net_reward = net_reward + r
    end

    --update Q
    Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
    s = sPrime

    if t % refresh == 0 then
        print(t,net_reward/refresh)
        gnuplot.plot(C:sum(2))
        print(Q)
        if net_reward/refresh > ((1/num_state)*(1-epsilon)) then
            break
        end
        net_reward = 0
    end

end




