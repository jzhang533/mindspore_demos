import numpy as np
from mindspore import dataset as ds
from mindspore.common.initializer import Normal
from mindspore import nn
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.common.parameter import ParameterTuple
import mindspore.ops as ops
from mindspore import context
from mindspore.train.dataset_helper import DatasetHelper, connect_network_with_dataset

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

def get_data(num, w=2.0, b=3.0):
    for i in range(num):
        x = np.random.uniform(-10.0, 10.0)
        noise = np.random.normal(0, 1)
        y = x * w + b + noise
        yield np.array([x]).astype(np.float32), np.array([y]).astype(np.float32)

def create_dataset(num_data, batch_size=16, repeat_size=1):
    input_data = ds.GeneratorDataset(list(get_data(num_data)), column_names=['data','label'])
    input_data = input_data.batch(batch_size)
    input_data = input_data.repeat(repeat_size)
    return input_data

class LinearNet(nn.Cell):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.fc = nn.Dense(1, 1, Normal(0.02), Normal(0.02))

    def construct(self, x):
        x = self.fc(x)
        return x

if __name__ == "__main__":
    num_data, batch_size, repeat_size = 1600, 16, 1
    lr, momentum = 0.005, 0.9
    
    network = LinearNet()
    net_loss = nn.loss.MSELoss()
    net_opt = nn.Momentum(network.trainable_params(), lr, momentum)

    net = WithLossCell(network, net_loss)
    net = TrainOneStepCell(net, net_opt)

    ds_train = create_dataset(num_data, batch_size=batch_size, repeat_size=repeat_size) 
    dataset_helper = DatasetHelper(ds_train, dataset_sink_mode=False, sink_size=100, epoch_num=10)

# dataset_sink_mode is not supported in CPU device 
#    dataset_helper = DatasetHelper(ds_train, dataset_sink_mode=True, sink_size=100, epoch_num=10)
#    net = connect_network_with_dataset(net, dataset_helper)

    network.set_train()
    print("============== Starting Training ==============")
    epoch = 2
    # a customized training loop
    for step in range(epoch):
        for inputs in dataset_helper:
            output = net(*inputs)
            print("epoch: {0}/{1}, losses: {2}".format(step + 1, epoch, output.asnumpy(), flush=True))

    print(network.trainable_params()[0], "\n%s" % network.trainable_params()[1])

