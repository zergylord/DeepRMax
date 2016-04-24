local eps = 1e-12
BCE = function(o,t)
    local loss = -( torch.cmul(t,torch.log(o+eps))+torch.cmul(-t+1,torch.log(-o+1+eps)) )
    return loss:sum(2):div(o:size(2))
end
