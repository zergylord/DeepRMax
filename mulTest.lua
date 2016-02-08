timer = torch.Timer()
refresh = 1e3
for i=1,1e10 do
    a = torch.rand(100,100)
    b = torch.rand(100,100)
    c = a*b
    if i % refresh == 0 then
        print(timer:time().real)
        timer:reset()
    end
end
