require 'gnuplot'
for i=1,10 do
    gnuplot.raw('set multiplot layout 3,2')
    --gnuplot.axis{0,10,0,.5}
    gnuplot.raw('set xrange [0:10]')
    gnuplot.raw('set yrange [0:.5] noreverse')
    gnuplot.plot(torch.rand(100))
    gnuplot.plot(torch.rand(100))
    gnuplot.imagesc(torch.rand(100,100))
    gnuplot.raw('unset multiplot')
    sys.sleep(1)
end
