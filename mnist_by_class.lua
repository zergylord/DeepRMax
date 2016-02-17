require 'hdf5'
f = hdf5.open('mnist.hdf5')
mnist_data = f:read():all()
digit = {}
in_dim = mnist_data.x_train:size(2)
for i=1,10 do
    mask = mnist_data.t_train:ne(i):reshape(50000,1):expandAs(mnist_data.x_train)
    local digit_data = mnist_data.x_train[mask]
    digit[i] = digit_data:reshape(digit_data:size(1)/in_dim,in_dim):double()
end
torch.save('digit.t7',digit)
