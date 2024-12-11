# CNN - Spin-glass correspondence

An investigation of the correspondence between convolutional neural networks and spin glasses.
The correspondence can be used to train CNNs using simulated or quantum annealing instead of backpropagation.

See [Neural Network - Spin Glass correspondence.pdf](./Neural%20Network%20-%20Spin%20Glass%20correspondence.pdf) for details.

The test code requires pytorch and the dwave-system api (`pip install dwave-system`).

To run the tests, first set the desired configuration in [main.py](./cnn/main.py) (network, dataset, training method,...):

```python
net, net_backward = networks[10]

#dataset
datasets=[MNIST, FashionMNIST, CIFAR10]
DATASET=datasets[0]
#get nr of input channels
in_channel = 3 if DATASET == CIFAR10 else 1

#training approach
trainings=["backpropagation", "simulated annealing", "quantum annealing"]
training=trainings[1]

#training configuration
N_epochs=20 #training epochs
N_samples=1000 #number of samples to use for training 
bsize=1 #batch size (used by backpropagation)
alpha=1 #LeakyReLU negative slope
```

Further network specific parameters can be set in the configuration.json file contained in the network folder in [saved models/](./cnn/saved_models/).
Accuracies for the test run are logged in the same folder.
