ReplayTable = {}
function ReplayTable.init(state_dim,num_frames,byte_storage)
    local D = {}
    D.num_frames = num_frames
    D.byte_storage = byte_storage
    D.state_dim = state_dim
    D.size = 1e6
    D.buf_size = 1024
    D.a = torch.zeros(D.size)
    D.r = torch.zeros(D.size)
    D.term = torch.zeros(D.size)
    if D.byte_storage then
        D.s = torch.ByteTensor(D.size,state_dim)
        D.buf_s = torch.Tensor(D.buf_size,state_dim*D.num_frames):cuda()
        D.buf_sPrime = torch.Tensor(D.buf_size,state_dim*D.num_frames):cuda()
        D.buf_i = D.buf_size+1
        D.buf_a = torch.zeros(D.size)
        D.buf_r = torch.zeros(D.size)
        D.buf_term = torch.zeros(D.size)
    else
        D.s = torch.zeros(D.size,state_dim)--s digit
    end
    D.i = 1
    D.fill = 0
    D.recent_past = torch.zeros(D.num_frames-1,D.state_dim)
    function D.add(D,s,a,r,term)
        if D.num_frames > 1 then
            if term then
                D.recent_past:zero()
            else
                D.recent_past[{{1,-2}}]:copy(D.recent_past[{{2,-1}}])
                D.recent_past[-1]:copy(s)
            end
        end
        if D.byte_storage then
            D.s[D.i] = s:mul(255):byte()
        else
            D.s[D.i] = s
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
            local s = torch.Tensor(mb_dim,D.state_dim)
            local sPrime = torch.Tensor(mb_dim,D.state_dim)
            local a = torch.Tensor(mb_dim)
            local r = torch.Tensor(mb_dim)
            local term = torch.Tensor(mb_dim)
            for i=1,mb_dim do
                s[i],sPrime[i],a[i],r[i],term[i] = D:sample_one()
            end
            return s:cuda(),a,r,sPrime:cuda(),term
        end
    end
    --Present frame is rightmost
    function D.get_future(D,ind)
        local frames = torch.ByteTensor(D.num_frames*D.state_dim):fill(0)
        local zero = false
        for i=D.num_frames,1,-1 do
            if not zero then
                frames[{{(i-1)*D.state_dim+1,i*D.state_dim}}] = D.s[ind+i-1]
            end
            if i<D.num_frames and D.term[ind+i-1] == 1 then
                zero = true
            end
        end
        return frames
    end
    function D.get_past(D,s)
        local full_s = D.recent_past:view(D.recent_past:numel()):cat(s)
        return full_s
    end
    --[[
    --get the predictable part of the state
    --]]
    function D.get_pred_state(s)
        if s:dim() == 1 then
            return s[{{D.state_dim*(D.num_frames-1),-1}}]
        else
            return s[{{},{D.state_dim*(D.num_frames-1),-1}}]
        end
    end
    function D.sample_one(D)
        local ind = torch.random(D.fill-D.num_frames+1)
        return D:get_future(ind),D:get_future(ind+1),D.a[ind],D.r[ind],D.term[ind]
    end


    return D
end

return ReplayTable
