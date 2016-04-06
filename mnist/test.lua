log2 = function(x) return torch.log(x)/torch.log(2) end
H = function(p) return log2(p)*(-p)-log2(-p+1)*(-p+1) end
