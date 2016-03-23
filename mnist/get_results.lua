res = torch.zeros(10)
for i = 1,10 do
    dofile('combolock.lua')
    res[i] = final_time
    print(res)
    torch.save('results.t7',res)
end

