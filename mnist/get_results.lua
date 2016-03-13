res = torch.zeros(10)
for i = 1,10 do
    dofile('no_count_er_maze.lua')
    res[i] = final_time
    print(res)
    torch.save('results.t7',res)
end

