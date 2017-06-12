## Hulk Smash Thor: Indoor Scenes Visual Navigation with A3C & Imitation Learning

![Hulk Smash Thor](https://68.media.tumblr.com/54028aee8ebfcbfa4817f7c87f2c5d90/tumblr_nnya5wXVXs1sss508o1_500.gif "Hulk Smash Thor")

## Introduction

This repository provides TensorFlow implementation of 2 extensions to the the [deep Siamese target-driven actor-critic model](http://web.stanford.edu/~yukez/icra2017.html) for indoor scenes navigation, first proposed by [Yuke Zhu](http://web.stanford.edu/~yukez/).

## Setup
This code was implemented in [Tensorflow r1.0](https://www.tensorflow.org/api_docs/). This code has been tested with Python 3.6. Other dependencies can be install with [pip](https://pypi.python.org/pypi/pip): `pip install -r requirements.txt`.

## Training and Evaluation
We include implementation for two separate models: A3C Target-driven with LSTM memory extension, and DAgger imitation learning-based model. The implementation of the former is under [train.py](train.py) and the class ActorCriticLSTMNetwork in [network.py](network.py). The implementation of the latter is located in [dagger_train.py](dagger_train.py) and [dagger_network.py](dagger_network.py).

To train a A3C-LSTM model, first activate the `USE_LSTM` tag in [constants.py](constants.py), then
```bash
python train.py
```

For the DAgger imitation-learning model,
```bash
python dagger_train.py
```

## Acknowledgements
We would like to acknowledge the following references that have offered great help in the implementation.
* [miyosuda's async_deep_reinforce repo](https://github.com/miyosuda/async_deep_reinforce)
* yukezhu's implementation of the Target-driven model (currently unavailable online)

## Citation
**[Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning](http://web.stanford.edu/~yukez/papers/icra2017.pdf)**
<br>
[Yuke Zhu](http://web.stanford.edu/~yukez/), Roozbeh Mottaghi, Eric Kolve, Joseph J. Lim, Abhinav Gupta, Li Fei-Fei, and Ali Farhadi
<br>
[ICRA 2017, Singapore](http://www.icra2017.org/)

## License
MIT
