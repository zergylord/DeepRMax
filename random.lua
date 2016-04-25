require 'gnuplot'
act_dim = 2
num_state = 30
e = 0
rmax = 1
alpha = .1
gamma = .5
Q = torch.zeros(num_state,act_dim)
visits = torch.zeros(num_state,act_dim)
T = torch.ones(num_state,act_dim)
correct = torch.ones(num_state)
for i = 1,num_state-1 do
    action = torch.random(act_dim)
    T[i][action] = i+1
    correct[i] = action
end
D = {}
D.size = 1e5
D.s = torch.zeros(D.size)
D.a = torch.zeros(D.size)
D.sPrime = torch.zeros(D.size)
D.count = 0

steps_per_update = 100
for epoch = 1,D.size/steps_per_update do
    --alpha = 1/epoch
    s = 1
    for i=1,steps_per_update do
        if e > torch.rand(1)[1] then
            a = torch.random(act_dim)
        else
            _,a = Q[s]:max(1)
            a = a[1]
        end
        visits[s][a] = visits[s][a] + 1
        D.count = D.count + 1
        D.s[D.count] = s
        D.a[D.count] = a
        D.sPrime[D.count] = T[s][a]
        s = T[s][a]
    end
    for t=1,1 do
        Q_copy = Q:clone()
        Q:zero()
        for i=1,D.count do
            s = D.s[i]
            sPrime = D.sPrime[i]
            for a = 1,act_dim do
                if a == D.a[i] then
                    Q[s][a] = (1-alpha)*Q[s][a]+ alpha*(gamma*Q_copy[sPrime]:max())
                else
                    if torch.rand(1)[1] < 1/act_dim then
                        Q[s][a] = (1-alpha)*Q[s][a]+ alpha*(rmax)
                    end
                end
            end
        end
    end
    gnuplot.figure(1)
    gnuplot.imagesc(visits)
    gnuplot.figure(2)
    gnuplot.imagesc(Q)
    visits:zero()
end
