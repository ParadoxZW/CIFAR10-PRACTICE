# CIFAR10 PRACTICE

Implementations of deep learning models to challenge cifar10 dataset, using PyTorch.
The implementation for network traing includes some kind of fancy tools,
like prefetch_generator, tqdm and tensorboardx.
I also use logging to print information into log file
rather than print function.
## Environment
- hardware: 2 GPUs
- Software: Pytorch, prefetch_generator, tensorboardX, tqdm

## Result
I only tested ResNet with 110 layers. I will keep update for better
performance and for other architechture.
Model | Test accuracy 
:-: | :-: 
ResNet110 | 0.84
ResNet110+Augmentation | 0.92 
## TODO
- [ ] Improve test accuracy
- [ ] Test other architecture