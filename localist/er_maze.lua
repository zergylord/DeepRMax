require 'gnuplot'
num_state = 10
Q = torch.zeros(num_state,4)
T = torch.ones(num_state,4)
correct = torch.ones(num_state)
for i = 1,num_state-1 do
    action = torch.random(4)
    T[i][action] = i+1
    correct[i] = action
end





visits = torch.zeros(num_state)
s = 1
epsilon = .1
alpha = .1
gamma = .9
net_reward = 0
refresh = 1e2
rmax = true
C = torch.zeros(num_state,4)
thresh = 10
D = {}
D.size = 1e6
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.r = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.i = 1
mb_dim = 32
for t=1,1e9 do
    r = 0
    --select action
    if rmax then
        _,a = torch.max(Q[s],1)
        a = a[1]
        --[[
        for i = 1,4 do
            if C[s][i] < thresh then
               a = i
               --r = (1-gamma)*1
            end
        end
        --]]
    elseif torch.rand(1)[1] < epsilon then
        a = torch.random(4)
    else
        _,a = torch.max(Q[s],1)
        a = a[1]
        --a = correct[s]
    end

    --perform action
    sPrime = T[s][a]
    visits[sPrime] = visits[sPrime] + 1


    C[s][a] = C[s][a] + 1
    if sPrime == num_state then
        r = 1
        net_reward = net_reward + r
    end
    --record history
    D.s[D.i] = s
    D.a[D.i] = a
    D.r[D.i] = r
    D.sPrime[D.i] = sPrime
    D.i = (D.i % D.size) + 1

    --update Q
    if t > mb_dim then
        mb_ind = torch.randperm(math.min(t,D.size))
        for i = 1,mb_dim do
            local s,a,r,sPrime
            s = D.s[mb_ind[i]]
            if rmax then
                --you can experience all actions under threshold, since they all go to heaven!
                for j = 1,4 do
                    a = j
                    if C[s][a] < thresh then
                        sPrime = s
                        --r = (1-gamma)*1
                        --Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
                        r = 1
                        Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*r
                    end
                end
                --only experienced actions can be updated over threshold
                a = D.a[mb_ind[i]]
                if C[s][a] >= thresh then
                    r = D.r[mb_ind[i]]
                    sPrime = D.sPrime[mb_ind[i]]
                    if s == num_state then
                        Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*r
                    else
                        Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
                    end
                end
            else
                r = D.r[mb_ind[i]]
                sPrime = D.sPrime[mb_ind[i]]
                a = D.a[mb_ind[i]]
                if s == num_state then
                    Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*r
                else
                    Q[s][a] = (1-alpha)*Q[s][a] + (alpha)*(r+gamma*torch.max(Q[sPrime]))
                end
            end
        end
    end
    s = sPrime

    if t % refresh == 0 then
        print(t,net_reward/refresh)
        --gnuplot.plot(C:sum(2))
        gnuplot.plot(Q:max(2):double())
        print(visits)
        visits:zero()
        if net_reward/refresh > ((1/num_state)*(1-epsilon)) then
            break
        end
        net_reward = 0
    end

end




