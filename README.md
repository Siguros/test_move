# IBM Analog Hardware Acceleration Kit

![PyPI](https://img.shields.io/pypi/v/aihwkit)
[![Documentation Status](https://readthedocs.org/projects/aihwkit/badge/?version=latest)](https://aihwkit.readthedocs.io/en/latest/?badge=latest)
![PyPI - License](https://img.shields.io/pypi/l/aihwkit)
[![arXiv](https://img.shields.io/badge/arXiv-2104.02184-green.svg)](https://arxiv.org/abs/2104.02184)

## Description

_IBM Analog Hardware Acceleration Kit_ is an open source Python toolkit for
exploring and using the capabilities of in-memory computing devices in the
context of artificial intelligence.

> :warning: This library is currently in beta and under active development.
> Please be mindful of potential issues and keep an eye for improvements,
> new features and bug fixes in upcoming versions.

The toolkit consists of two main components:

### Pytorch integration

A series of primitives and features that allow using the toolkit within
[`PyTorch`]:

* Analog neural network modules (fully connected layer, 1d/2d/3d convolution
  layers, LSTM layer, sequential container).
* Analog training using torch training workflow:
  * Analog torch optimizers (SGD).
  * Analog in-situ training using customizable device models and algorithms
    (Tiki-Taka).
* Analog inference using torch inference workflow:
  * State-of-the-art statistical model of a phase-change memory (PCM) array
    calibrated on hardware measurements from a 1 million PCM devices chip.
  * Hardware-aware training with hardware non-idealities and noise
    included in the forward pass to make the trained models more
    robust during inference on Analog hardware.

### Analog devices simulator

A high-performant (CUDA-capable) C++ simulator that allows for
simulating a wide range of analog devices and crossbar configurations
by using abstract functional models of material characteristics with
adjustable parameters. Features include:

* Forward pass output-referred noise and device fluctuations, as well
  as adjustable ADC and DAC discretization and bounds
* Stochastic update pulse trains for rows and columns with finite
  weight update size per pulse coincidence
* Device-to-device systematic variations, cycle-to-cycle noise and
  adjustable asymmetry during analog update
* Adjustable device behavior for exploration of material specifications for
  training and inference
* State-of-the-art dynamic input scaling, bound management, and update
  management schemes

### Other features

Along with the two main components, the toolkit includes other
functionalities such as:

* A library of device presets that are calibrated to real hardware data and
  based on models in the literature, along with a configuration that specifies a particular device and optimizer choice.
* A module for executing high-level use cases ("experiments"), such as neural
  network training with minimal code overhead.
* A utility to automatically convert a downloaded model (e.g., pre-trained) to its equivalent Analog
  model by replacing all linear/conv layers to Analog layers (e.g., for convenient hardware-aware training).
* Integration with the [AIHW Composer] platform, a no-code web experience that allows executing
  experiments in the cloud.

## How to cite?

In case you are using the _IBM Analog Hardware Acceleration Kit_ for
your research, please cite the AICAS21 paper that describes the toolkit:

> Malte J. Rasch, Diego Moreda, Tayfun Gokmen, Manuel Le Gallo, Fabio Carta,
> Cindy Goldberg, Kaoutar El Maghraoui, Abu Sebastian, Vijay Narayanan.
> ["A flexible and fast PyTorch toolkit for simulating training and inference on
> analog crossbar arrays"](https://ieeexplore.ieee.org/abstract/document/9458494) (2021 IEEE 3rd International Conference on Artificial Intelligence Circuits and Systems)
>

## Usage

### Training example

```python
from torch import Tensor
from torch.nn.functional import mse_loss

# Import the aihwkit constructs.
from aihwkit.nn import AnalogLinear
from aihwkit.optim import AnalogSGD

x = Tensor([[0.1, 0.2, 0.4, 0.3], [0.2, 0.1, 0.1, 0.3]])
y = Tensor([[1.0, 0.5], [0.7, 0.3]])

# Define a network using a single Analog layer.
model = AnalogLinear(4, 2)

# Use the analog-aware stochastic gradient descent optimizer.
opt = AnalogSGD(model.parameters(), lr=0.1)
opt.regroup_param_groups(model)

# Train the network.
for epoch in range(10):
    pred = model(x)
    loss = mse_loss(pred, y)
    loss.backward()

    opt.step()
    print('Loss error: {:.16f}'.format(loss))
```

You can find more examples in the [`examples/`] folder of the project, and
more information about the library in the [documentation]. Please note that
the examples have some additional dependencies - you can install them via
`pip install -r requirements-examples.txt`.
You can find interactive notebooks and tutorials in the [`notebooks/`] directory.

### Further reading

We also recommend to take a look at the tutorial article that
describes the usage of the toolkit that can be found here:

> Manuel Le Gallo, Corey Lammie, Julian Buechel, Fabio Carta, Omobayode Fagbohungbe,
> Charles Mackin, Hsinyu Tsai, Vijay Narayanan, Abu Sebastian, Kaoutar El Maghraoui,
> Malte J. Rasch.
> ["Using the IBM Analog In-Memory Hardware Acceleration Kit for Neural Network Training and Inference"](https://doi.org/10.1063/5.0168089)
> (APL Machine Learning Journal:1(4) 2023)
>


## What is Analog AI?

In traditional hardware architecture, computation and memory are siloed in
different locations. Information is moved back and forth between computation
and memory units every time an operation is performed, creating a limitation
called the [von Neumann bottleneck].

Analog AI delivers radical performance improvements by combining compute and
memory in a single device, eliminating the von Neumann bottleneck. By leveraging
the physical properties of memory devices, computation happens at the same place
where the data is stored. Such in-memory computing hardware increases the speed
and energy efficiency needed for next-generation AI workloads.

## What is an in-memory computing chip?

An in-memory computing chip typically consists of multiple arrays of memory
devices that communicate with each other. Many types of memory devices such as
[phase-change memory] (PCM), [resistive random-access memory] (RRAM), and
[Flash memory] can be used for in-memory computing.

Memory devices have the ability to store synaptic weights in their analog
charge (Flash) or conductance (PCM, RRAM) state. When these devices are arranged
in a crossbar configuration, it allows to perform an analog matrix-vector
multiplication in a single time step, exploiting the advantages of analog
storage capability and [Kirchhoff’s circuits laws]. You can learn more about
it in our [online demo].

In deep learning, data propagation through multiple layers of a neural network
involves a sequence of matrix multiplications, as each layer can be represented
as a matrix of synaptic weights. The devices are arranged in multiple crossbar
arrays, creating an artificial neural network where all matrix multiplications
are performed in-place in an analog manner. This structure allows to run deep
learning models at reduced energy consumption.

## Awards and Media Mentions

* IBM Research blog: [Open-sourcing analog AI simulation]: <https://research.ibm.com/blog/analog-ai-for-efficient-computing>
* We are proud to share that the AIHWKIT and the companion cloud composer received the IEEE OPEN SOURCE SCIENCE [award] in 2023.
  <img width="817" alt="IEEE Award" src="https://github.com/IBM/aihwkit/assets/7916630/bd3347fe-49c0-4aa2-ba6e-455b0e2a91ce">

## Installation

### Installing the Custom LR-TT CUDA Build

This repository contains a custom CUDA-enabled build of AIHWKit with LR-TT (Low-Rank Tensor Train) support. For installation instructions for this specific build, see the [LR-TT Installation Guide](#lrtt-installation-guide) section below.

### Installing from PyPI (Standard Version)

The preferred way to install the standard package is by using the
[Python package index]:

```shell
pip install aihwkit
```

### Conda-based Installation
There is a conda package for aihwkit available in conda-forge.   It can be  installed in a conda environment running on a Linux or WSL in a Windows system.  

* CPU

  ```shell
  conda install -c conda-forge aihwkit
  ```

* GPU 

  ```shell
  conda install -c conda-forge aihwkit-gpu
  ```

If you encounter any issues during download or want to compile the package
for your environment, please take a look at the [advanced installation] guide.
That section describes the additional libraries and tools required for
compiling the sources using a build system based on `cmake`.

### Docker Installation 

For GPU support, you can also build a docker container following the [CUDA Dockerfile instructions].
You can then run a GPU enabled docker container using the follwing command from your peoject dircetory

```shell
docker run --rm -it --gpus all -v $(pwd):$HOME --name aihwkit aihwkit:cuda bash
```

## LR-TT Installation Guide

This repository contains a custom CUDA-enabled build of AIHWKit with LR-TT (Low-Rank Tensor Train) support specifically built for CUDA 12.1 and Python 3.10.

### Quick Install

Download and install the wheel directly from the GitHub release:

```bash
# Download the wheel
curl -L -o aihwkit-lrtt.whl https://github.com/Siguros/test_move/releases/download/v1.0.0-lrtt/aihwkit-0.9.0-cp310-cp310-linux_x86_64.whl

# Install with no dependencies (recommended)
pip install --no-deps ./aihwkit-lrtt.whl
```

### Install via Asset ID (for private repos)

If you have a GitHub token with access to private repositories:

```bash
# Set your GitHub token
export GITHUB_TOKEN="your_github_token_here"

# Download via asset ID (avoids filename encoding issues)
curl -L -H "Authorization: token ${GITHUB_TOKEN}" \
     -H "Accept: application/octet-stream" \
     "https://api.github.com/repos/Siguros/test_move/releases/assets/285490233" \
     -o aihwkit-lrtt.whl

# Install with no dependencies
pip install --no-deps ./aihwkit-lrtt.whl
```

### Verification

After installation, verify that CUDA support and LR-TT functionality work:

```python
import aihwkit
from aihwkit.simulator.rpu_base import cuda
import torch

print("AIHWKit version:", aihwkit.__version__)
print("CUDA compiled:", cuda.is_compiled())

# Test LR-TT functionality
from aihwkit.nn import AnalogLinear
from aihwkit.simulator.presets.lrtt import lrtt_idealized

# Create LR-TT configuration
cfg = lrtt_idealized(rank=16)
cfg.device.forward_inject = True
cfg.device.lora_alpha = 8.0

# Create analog layer
layer = AnalogLinear(32, 16, bias=False, rpu_config=cfg)

# Move to CUDA if available
if torch.cuda.is_available():
    layer = layer.cuda()
    x = torch.randn(4, 32, device='cuda')
    print("✓ Using CUDA device")
else:
    x = torch.randn(4, 32)
    print("! Using CPU device")

# Test forward and backward pass
y = layer(x)
loss = y.sum()
loss.backward()

print("✓ LR-TT forward/backward pass successful")
print("Output shape:", y.shape)
```

### Requirements for LR-TT Build

- **Python**: 3.10
- **CUDA**: 12.1 (cu121)
- **PyTorch**: Compatible with CUDA 12.1
- **Platform**: Linux x86_64

### Build Information

This wheel was built with:
- **CUDA Support**: Enabled (CUDA 12.1)
- **LR-TT Support**: Enabled
- **Platform**: linux_x86_64
- **Python**: 3.10

### Troubleshooting LR-TT Build

If `cuda.is_compiled()` returns `False`:

1. Ensure you have CUDA 12.1 installed
2. Verify PyTorch was installed with CUDA support:
   ```python
   import torch
   print("PyTorch CUDA available:", torch.cuda.is_available())
   print("PyTorch CUDA version:", torch.version.cuda)
   ```
3. Check that you're using Python 3.10

For installation issues:
1. Use `--no-deps` flag to avoid dependency conflicts
2. Ensure wheel compatibility (Python 3.10, Linux x86_64)
3. Verify CUDA 12.1 is properly installed

## Authors

IBM Research has developed IBM Analog Hardware Acceleration Kit,
with Malte Rasch, Diego Moreda, Fabio Carta, Julian Büchel, Corey Lammie, Charles Mackin, Kim Tran, Tayfun Gokmen, Manuel Le Gallo-Bourdeau, and Kaoutar El Maghraoui
as the initial core authors, along with many [contributors].

You can contact us by opening a new issue in the repository or alternatively
at the ``aihwkit@us.ibm.com`` email address.

## License

This project is licensed under [MIT License].

[MIT License 2.0]: LICENSE.txt
[Python package index]: https://pypi.org/project/aihwkit
[`PyTorch`]: https://pytorch.org/

[`examples/`]: examples/
[`notebooks/`]: notebooks/
[documentation]: https://aihwkit.readthedocs.io/
[contributors]: https://github.com/IBM/aihwkit/graphs/contributors
[advanced installation]: https://aihwkit.readthedocs.io/en/latest/advanced_install.html

[von Neumann bottleneck]: https://en.wikipedia.org/wiki/Von_Neumann_architecture#Von_Neumann_bottleneck
[phase-change memory]: https://en.wikipedia.org/wiki/Phase-change_memory
[resistive random-access memory]: https://en.wikipedia.org/wiki/Resistive_random-access_memory
[Flash memory]: https://en.wikipedia.org/wiki/Flash_memory
[Kirchhoff’s circuits laws]: https://en.wikipedia.org/wiki/Kirchhoff%27s_circuit_laws
[online demo]: https://analog-ai-demo.mybluemix.net/
[AIHW Composer]: https://aihw-composer.draco.res.ibm.com
[award]: https://conferences.computer.org/services/2023/awards/
[CUDA Dockerfile instructions]: https://github.com/IBM/aihwkit/blob/master/docs/source/advanced_install.rst#cuda-enabled-docker-image
