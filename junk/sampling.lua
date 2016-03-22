require 'gnuplot'
require 'distributions'
--[[PDF viz
points = torch.rand(1000,2) --:mul(10):add(-5)
x = points[{{},{1}}]
y = points[{{},{2}}]
values = distributions.mvn.pdf(points,torch.zeros(2),torch.ones(2))

values2 = distributions.mvn.pdf(points,torch.rand(2):add(-.5):mul(5),torch.rand(2))
--gnuplot.splot(x,y,values)
gnuplot.scatter3({x[{{},1}],y[{{},1}],values[{{},1}]},{x[{{},1}],y[{{},1}],values2[{{},1}]})
--]]
--sampling viz
mu = torch.randn(2)
sigma = torch.randn(2,2)
values = distributions.mvn.rnd(torch.zeros(10000,2),mu,sigma)
gnuplot.axis{-10,10,-10,10}
gnuplot.plot(values,'+')

