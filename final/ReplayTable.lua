ReplayTable = {}
function ReplayTable.init(in_dim,byte_storage)
    local D = {}
    D.byte_storage = byte_storage
    D.in_dim = in_dim
    D.size = 1e6
    D.buf_size = 1024
    D.a = torch.zeros(D.size)
    D.r = torch.zeros(D.size)
    D.term = torch.zeros(D.size)
    if D.byte_storage then
        D.s = torch.ByteTensor(D.size,in_dim)
        D.buf_s = torch.Tensor(D.buf_size,in_dim):cuda()
        D.sPrime = torch.ByteTensor(D.size,in_dim)--sPrime digit
        D.buf_sPrime = torch.Tensor(D.buf_size,in_dim):cuda()
        D.buf_i = D.buf_size+1
        D.buf_a = torch.zeros(D.size)
        D.buf_r = torch.zeros(D.size)
        D.buf_term = torch.zeros(D.size)
    else
        D.s = torch.zeros(D.size,in_dim)--s digit
        D.sPrime = torch.zeros(D.size,in_dim)--sPrime digit
    end
    D.i = 1
    D.fill = 0
    function D.add(D,s,a,r,sPrime,term)
        if D.byte_storage then
            D.s[D.i] = s:mul(255):byte()
            D.sPrime[D.i] = sPrime:mul(255):byte()
        else
            D.s[D.i] = s
            D.sPrime[D.i] = sPrime
        end
        D.a[D.i] = a
        D.r[D.i] = r
        if term then
            D.term[D.i] = 1
        else
            D.term[D.i] = 0
        end
        D.i = (D.i % D.size) + 1
        if D.fill < D.size then
            D.fill = D.fill + 1
        end
    end
    function D.fill_buffer(D)
        D.buf_i = 1
        local s,sPrime
        for i =1,D.buf_size do
            s,sPrime,D.buf_a[i],D.buf_r[i],D.buf_term[i] = D:sample_one()
            D.buf_s[i] = s:float():div(255)
            D.buf_sPrime[i] = sPrime:float():div(255)
        end
    end

    function D.get_samples(D,mb_dim)
        if D.byte_storage then
            if D.buf_i+mb_dim-1 > D.buf_size then
                D:fill_buffer()
            end
            local range = {{D.buf_i,D.buf_i+mb_dim-1}}
            D.buf_i = D.buf_i + mb_dim
            return D.buf_s[range],D.buf_a[range],D.buf_r[range],D.buf_sPrime[range],D.buf_term[range]
        else
            local s = torch.Tensor(mb_dim,D.in_dim)
            local sPrime = torch.Tensor(mb_dim,D.in_dim)
            local a = torch.Tensor(mb_dim)
            local r = torch.Tensor(mb_dim)
            local term = torch.Tensor(mb_dim)
            for i=1,mb_dim do
                s[i],sPrime[i],a[i],r[i],term[i] = D:sample_one()
            end
            return s:cuda(),a,r,sPrime:cuda(),term
        end
    end
    function D.sample_one(D)
        local ind = torch.random(D.fill)
        return D.s[ind ],D.sPrime[ind],D.a[ind],D.r[ind],D.term[ind]
    end


    return D
end

return ReplayTable

