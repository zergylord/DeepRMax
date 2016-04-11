num_iter = 30
param = torch.zeros(num_iter)
score = torch.zeros(num_iter)
for i = 1,num_iter do
    temp = torch.rand(1):mul(2)[1]
    print(temp)
    dofile('combolock.lua')
    score[i] = final_time
    param[i] = temp
    print(final_time,temp)
    torch.save('results.t7',{param,score})
end

