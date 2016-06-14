*Stefan Lattner, Maarten Grachten, Carlos Eduardo Cancino Chacon*

### LRN2 - Framework (including the nn_bricks package)

This is a python package that provides tools for learning representations from data, implemented in the context of the 
[Lrn2Cre8.eu project](http://lrn2cre8.eu/ "Lrn2Cre8 website"). Its goal is to simplify the creation and training of Neural Network architectures using [theano](http://deeplearning.net/software/theano/), as well as reducing the effort in the associated data management.
It includes the **nn_bricks** package, providing classes which can be combined in a building block like manner making it easy to create custom Neural Network (NN) architectures. Pre-defined layers can be created, stacked and trained using a configuration file.

Among others, **lrn2** includes [Convolutional] *Restricted Boltzmann Machines (RBMs)*, [Convolutional] *Neural Networks and Auto-Encoders*, [Gated] *RNNs, LSTMs*, corresponding *training routines and cost functions*, as well as common *regularizers*. [Overview of available bricks](#nn-bricks).

The structure of **lrn2** is designed to facilitate the creation of an end-to-end work flow. This work flow involves the following four phases:

1. Reading and preparing data from files
2. Defining and creating models using a configuration file
3. Training models to learning representations from the data
4. Using the trained models in specific use case scenarios


Files are organized into several subdirectories (under **lrn2**):
* **nn_bricks**: This package provides brick-like classes (i.e. predefined NN, RNN and RBM layers, several unit types, cost functions, regularizers, training algorithms, plotting functions, serialization methods, ...) which can be freely combined into mix-in classes to define different NN layers or stacks.
* **data**: Data holding and data import. Interfaces for different domains and formats.
* **util**: Several utility modules specific for this framework
* **application**:  Code which can be used for special applications (like classification or visualization)

### Links
* [Documentation](http://lrn2cre8.ofai.at/lrn2/doc/)
* [Tutorials](https://github.com/OFAI/lrn2/tree/github_orphan/tutorials)
* [Project website](http://lrn2cre8.eu/?q=workpackage1/deliverable1.1)

### Installation
To install the lrn2 framework (Linux/Mac), run the following command in your terminal

```
git clone https://github.com/OFAI/lrn2.git
cd lrn2
git submodule init
git submodule update
sudo python [setup.py | setup_mac.py] install
```

or download the source and use it in your IDE (feel free to fork). Running the above command may still help to install the required dependencies. 

**Test the lrn2 framework:**

- In the (extracted) tar archive, browse to an example folder **examples/[example_name]**. Run one of the examples by entering
`python run_demo.py [any_run_keyword] config_model.ini [options]`, or with the `--help` flag for help on the specific example. For the **mnist_** examples, no further parameters are needed.
The other two examples need the path to a .csv file with (monophonic) melodies (e.g. the [Essen Folksong Collection](http://www.esac-data.org/)) as additional parameter. You can generate such a .csv file by using `python create_csv_filelist.py` in `lrn2/util`.
- If some import errors are shown, ensure that all dependencies are included in the environment variable $PYTHONPATH.

### Uninstall lrn2
Run the following command in your terminal: `sudo pip uninstall lrn2`

### Example
The example below shows the main workflow of the **lrn2** framework (omitting data preparation for now). 
`lrn2.nn_bricks.make.make_net` takes a configuration file and creates NN layers accordingly. The layers are then packed into a stack with cross-entropy cost (custom stacks and layers can be defined using some bricks of the *nn_bricks* collection). Binding data to variables (`input` and `target` are some default variables) is done using a dictionary. `lrn2.nn_bricks.train.train_cached` is one of the two training methods of the framework, which trains a whole stack or a single layer. The other, `lrn2.nn_bricks.train.train_layer_wise` could be used for e.g. greedy, layer-wise pretraining of an RBM stack or for layer-wise pre-training of an Auto-Encoder stack.
```python
# Create NN layers (according to a configuration)
layers = make_net(config, input_shape = (28, 28))

# Pack layers in stack with cross entropy cost
stack = FFNNCrossEntropy(layers, name = 'mnist_classifier')

# Define input data (mnist_data and mnist_labels are numpy arrays)
data = {'input': mnist_data, 'target': mnist_labels}

# Train the whole stack with backpropagation in discriminative task
train_cached(stack, data, config, run_keyword = 'mnist_classifier', re_train = True)
```
For an extended introduction, see the tutorial [first steps](https://github.com/OFAI/lrn2/blob/github/tutorials/tutorial01.ipynb).
### Config File
A config file (using the [ConfigObj](http://www.voidspace.org.uk/python/configobj.html) library) determines the nets creation and training parameters. In the case below, we create a three-layered **Convolutional Neural Network** with **Rectified Linear Units** and **Batch Normalization**, a **Max-Pooling** layer between the first and the second layer, and a dense layer with **Softmax** units on top (with 10 neurons for 10 MNIST classes). The category `[STACK]` defines some training parameters for the full stack. Training parameters defined in the layer categories are only related to layer-wise training (except regularizers like the *weight regularization* defined via `wl2`, which remain part of the resulting cost function).
```
[OUTPUT]
output_path = nn_output     # Resulting files will go nn_output/[run_keyword]

[INPUT]
ngram_size = 1              # 1 for images (e.g. MNIST)

[TOPOLOGY]
[[Layer1]]
type_net = NNConvReLUBN     # class name of the layer in lrn.nn_bricks.make
                            # (Convolutional NN with ReLU units and Batch Normalization)
convolutional = True        # convolutional flag
filter_shape = 5, 5         # shape of conv filter
n_hidden = 32               # hidden maps
wl2 = 1e-3                  # wl2 weight regularization

[[LayerPooling1]]
type_net = MaxPooling       # class name of the layer in lrn.nn_bricks.make
                            # (Max-pooling layer)
convolutional = True        # convolutional flag
downsample = 2, 2           # downsample by 1/2 in each dimension
no_param_layer = True       # layer has no params

[[Layer2]]
type_net = NNConvReLUBN     # class name of the layer in lrn.nn_bricks.make
                            # (Convolutional NN with ReLU units and Batch Normalization)
convolutional = True        # convolutional flag
filter_shape = 5, 5         # shape of conv filter
n_hidden = 32               # hidden maps
wl2 = 1e-3                  # wl2 weight regularization

[[LayerToDense]]
type_net = ConvToDense      # class name of the layer in lrn.nn_bricks.make
                            # (Convolution to dense transition layer)
convolutional = True        # convolutional flag
no_param_layer = True       # layer has no params

[[LayerTop]]
type_net = NNSoftmaxBN      # class name of the layer in lrn.nn_bricks.make
                            # Dense NN layer with softmax units and batch normalization
n_hidden = 10               # 10 hidden units
wl2 = 0.001                 # wl2 weight regularization

[STACK]
img_interval = 25           # Plot params, histograms, cost curves every 25 epochs
dump_interval = 50          # Backup model parameters every 50 epochs

learning_rate = 5e-3        # Learning rate
momentum = 0.85             # Momentum during training
batch_size = 100            # Mini-batch size
epochs = 151                # Number of epochs to train
reduce_lr = False           # Don`t reduce learning rate (to zero over epochs)
```
The file can be imported using the method `lrn2.util.config.get_config(config_file_path, specification_path)`, which returns a dictionary reflecting the config file content and structure. The `specification_path` (default is **lrn2/util/config_spec.ini**) points to a specification file, which defines the parameter defaults, variable types, and the config files structure. You can always add a new parameter (e.g. to be used with a custom layer) and use it immediately in a brick, as long as it receives **kwargs (kwargs['parameter_name']), however, if its variable type is not defined in **util/config_spec.ini**, it has to be cast to the desired type. In order to avoid casting, copy the **config_spec.ini** in your project folder, edit it accordingly and pass it in `specification_path` of `get_config`. 

Specific layer parameters are forwarded to the respective layer constructors as **kwargs and should be passed on to most mix-in class constructors (those who take parameters at all). Parameters defined in the config file but not used by any mix-in class (e.g. `wl2` is defined in config file, but the corresponding layer does not derive from `WeightRegular`) are ignored.

For further information on the configuration file, see [Tutorial 3](https://github.com/OFAI/lrn2/blob/github/tutorials/tutorial03.ipynb), [Tutorial 1](https://github.com/OFAI/lrn2/blob/github/tutorials/tutorial02.ipynb), or see the related [Documentation](http://lrn2cre8.ofai.at/lrn2/doc/lrn2.util.html#module-lrn2.util.config).

### Data handling
**lrn2** provides two classes for data management, the `lrn2.data.Corpus` class and the `lrn2.data.LiveCorpus` class. The `Corpus` class should be used for data which fits in the working memory, while the `LiveCorpus` is made for managing big data, which is stored and loaded batch by batch from a possibly fast (i.e. SSD) hard disk. However, both classes provide a very similar interface and their use do not essentially differ from each other. Note that data handling is fully independent from creation and training of the models, the final data fed into the models (`corpus.ngram_data`) is a simple numpy array.

**Example**
```python
corpus = Corpus(file_loader = load_mnist_files, viewpoints = [MnistVP(shape = (28, 28))])
corpus.load_files(datafiles)        # calls load_mnist_files(datafiles) for raw data and passes it to
                                    # mnist_vp.raw_to_repr(raw_data) to obtain input representation
corpus.set_to_ngram(ngram_size = 1) # Prepares the actual data, here each row is a linearized image
corpus.shuffle_instances()          # Shuffle rows
data = corpus.ngram_data            # 2d numpy array: [instance_count, linearized_image_size]
```
The formalization consists in the separation of the input data processing into two components:

1. **data loading function** a function for loading data from files (*format specific*, e.g. **MIDI**)
2. **viewpoints** a set of viewpoint objects that produce a vectorized representation of some aspect of the data returned by the data loading function, in the form of one or more instances (*domain specific*, e.g. **Music**)

When `corpus.load_files(datafiles)` is called, `datafiles` is directly passed on to the file loader (here: `load_mnist_files`), which reads the files and returns the **raw data**. The viewpoint (here: `MnistVP`) receives this raw data and turns it into a **representation**, suited for an input to a Neural Network. For more information on data handling, see the related [Tutorial](https://github.com/OFAI/lrn2/blob/github/tutorials/tutorial02.ipynb), refer to the [Project website](http://lrn2cre8.eu/?q=workpackage1/deliverable1.1), the [Documentation](http://lrn2cre8.ofai.at/lrn2/doc/), or inspect some examples in the /examples folder.

### NN Bricks
The idea of nn_bricks is to write once and reuse NN building blocks, constituting typical components of NNs. In the following, find an overview of some pre-defined bricks in **lrn2**:

* **[layers](http://lrn2cre8.ofai.at/lrn2/doc/lrn2.nn_bricks.html#module-lrn2.nn_bricks.layers)**: `NN` (Standard Neural Network layer), `NN_BN` (NN with batch normalization), `NNAuto` (Autoencoder layer), `CNN` (Convolutional NN layer), `CNN_BN` (Convolutional NN with batch normalization), `DCNN`(de-convolutional NN layer), `RBM` (Restricted Boltzmann Machine), `CRBM` (Convolutional Restricted Boltzmann Machine), `CondCRBM` (Conditional Convolutional Restricted Boltzmann Machine), `RNN` (Recurrent Neural Network), `RNN_Gated` (Gated RNN), `LSTM` (Long-short-term Memory), `ToDense` (Convolutional to dense layer), `ConvShaping`, `MaxPooler`, ...
 
* **[units](http://lrn2cre8.ofai.at/lrn2/doc/lrn2.nn_bricks.html#module-lrn2.nn_bricks.units)**: `UnitsNNSigmoid`, `UnitsNNTanh`, `UnitsNNLinear`, `UnitsNNLinearNonNeg` `UnitsNNSoftmax`, `UnitsNNReLU`, `UnitsRBMSigmoid`, `UnitsRBMReLU`, `UnitsRBMGauss`, `UnitsNoisy` (adds noise before activation function), `UnitsDropOut` (Adds dropout to any unit type), ...

* **[cost](http://lrn2cre8.ofai.at/lrn2/doc/lrn2.nn_bricks.html#module-lrn2.nn_bricks.cost)**: `CostCrossEntropy`, `CostCategoricCrossEntropy`, `CostKL` (Kullback-Leibler cost), `CostMaxLikelihood`, `CostSquaredError`, `CostReconErr`, `CostCD` (Contrastive Divergence cost), `CostPCD` (Persistent Contrastive Divergence cost), `CostPCDFW` (PCD cost with fast weights), `CostRNNPred` (Cross Entropy cost for prediction RNN, no need to define target data (as target = roll(input, -1)), ...
 
* **[regularizer](http://lrn2cre8.ofai.at/lrn2/doc/lrn2.nn_bricks.html#module-lrn2.nn_bricks.regularize)**: `MaxNormRegular` (Max-norm regularization), `SparsityLee` (Lee et. al. sparsity regularizer), `SparsityLeeConv` (for convolutional layers), `WeightRegular` (w1/w2 weight regularization), `SparsityGoh` (Goh et. al. sparsity regularization), `SparsityGOHConv` (for convolutional layers), `NonNegative` (prevents negative weights), `ActivationCrop` (shuffles weights of neurons which are too active), ...

Those bricks are then used to get mixed in a class, constituting a layer or a stack (as both derive from `FFBase`, they share some properties, like input / output, cost, notifier,..), for example

```python
class NNSigmoidBN(Notifier, UnitsNNSigmoid, UnitsDropOut, NN_BN, CostCrossEntropy, WeightRegular,
                  MaxNormRegular, SparsityLee, SerializeLayer, Monitor, Plotter):
    """ 
    NN Layer with sigmoid activation and batch normalization.
    """ 
    def __init__(self, **kwargs):
        kwargs['variables'] = create_in_trg(kwargs['name'])
        Notifier.__init__(self)                     # Notifier should be always added
        UnitsNNSigmoid.__init__(self)               # Sigmoid units
        UnitsDropOut.__init__(self, **kwargs)       # ..with Dropout
        NN_BN.__init__(self, **kwargs)              # Standard NN layer with batch normalization
        CostCrossEntropy.__init__(self, **kwargs)   # Cross Entropy cost
        WeightRegular.__init__(self, **kwargs)      # Weight regularization
        MaxNormRegular.__init__(self, **kwargs)     # Max-norm regularization
        SparsityLee.__init__(self, **kwargs)        # Lee sparsity regularization
        SerializeLayer.__init__(self)               # Adds ability to serialize layer
        Monitor.__init__(self)                      # Adds ability to monitor the training
        self.notify(Notifier.MAKE_FINISHED)         # Those three notifications are recommended after...
        self.notify(Notifier.COMPILE_FUNCTION)      # ...initializing all mix-in classes.
        self.notify(Notifier.REGISTER_PLOTTING)
```
The **order** of calling the respective constructors should always be the following: the notifier, units, the actual layer logic, the cost function, regularizers, the serializer and the monitor. Finally, notifications should be sent, as in the example above. The constructor is called by the `make_net` method, which passes all configuration parameters of the corresponding section of the config file in kwargs, as well as other related information.

[Several such predefined layers are available](http://lrn2cre8.ofai.at/lrn2/doc/lrn2.nn_bricks.html#module-lrn2.nn_bricks.make) and can be used by defining them in a config file: `type_net = [class name]`. Custom layers can be defined, too (in that case use `type_net = custom` and the additional property `custom_type = [class name]`). For an example using custom layers, see **example/custom_layer**.

### Stacks
Layers created by `make_net` are not connected. In order to connect them, initialize a stack like
```python
# Create NN layers (according to a configuration)
layers = make_net(config, input_shape = (28, 28))
# Pack layers in stack with cross entropy cost
stack = FFNNCrossEntropy(layers, name = 'mnist_classifier')
```
A list and description of pre-defined stacks in `lrn2.nn_bricks.stacks` can be found [here](http://lrn2cre8.ofai.at/lrn2/doc/lrn2.nn_bricks.html#module-lrn2.nn_bricks.stacks).

Stacks are build from mix-in classes, like single layers. In fact, they derive (indirectly) from `FFBase`, as they have things in common with single layers (e.g. symbolic input, symbolic target, cost function, variables, data binding and plot functions). Most importantly, stacks derive from `NNStack`, which is initialized with a list of layers. An `NNStack` creates a theano graph by connecting outputs with inputs, builds a global cost function by collecting regularizers of the layers, collects registered notifiers, and so on. Note that the current `NNStack` assumes layers with one input and one output each. For more complicated architectures, you can either encapsulate the complexity in custom (single) layers, or extend the `NNStack` class according to your needs.
```python
class FFNNCrossEntropy(Notifier, NNStack, CostCrossEntropy, SerializeStack, Monitor, Plotter):
    """ *Stack:* Feed forward neural network with cross entropy cost """
    def __init__(self, layers, name, **kwargs):
        Notifier.__init__(self)
        NNStack.__init__(self, layers, layers[0].input, name)
        CostCrossEntropy.__init__(self, **kwargs)
        SerializeStack.__init__(self)
        Monitor.__init__(self)
        Plotter.__init__(self)
        self.notify(Notifier.COMPILE_FUNCTIONS)
        self.notify(Notifier.REGISTER_PLOTTING)
```
Example of a feed-forward NN stack with cross-entropy cost as a combination of some mix-in classes.
#### Acknowledgments
The project Lrn2Cre8 acknowledges the financial support of the  Future  and  Emerging  Technologies  (FET)  programme within the Seventh Framework Programme for Research of the European Commission, under FET grant number 610859.
