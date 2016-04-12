num_iter = 300
param = torch.zeros(num_iter)
score = torch.zeros(num_iter)
for i = 1,num_iter do
    --local val = torch.rand(1):mul(2)[1]
    --distributions.cat.rnd(1,torch.ones(2),{categories=torch.Tensor{1}})
    val = torch.rand(1)[1]*10^-2
    print(val)
    dofile('combolock.lua')
    score[i] = final_time
    param[i] = val
    print(final_time,val)
    torch.save('results.t7',{param,score})
end

