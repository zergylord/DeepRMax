require 'gnuplot'
f = function(x,c) return torch.cdiv(x,x+c) end
fPrime = function(x,c) return -torch.cdiv(x,torch.pow(x+c,2)) + (x+c):pow(-1) end
x = torch.linspace(0,10)
gnuplot.figure(1)
gnuplot.axis{'','',0,1}
for i=1,10 do
c = .25*2^(i-1)
gnuplot.plot({x,f(x,c)},{x,fPrime(x,c)})
sys.sleep(1)
end

