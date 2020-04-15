# CIFAR10 PRACTICE

Implementations of deep learning models to challenge cifar10 dataset, using PyTorch.
The implementation for network traing includes some kind of fancy tools,
like prefetch_generator, tqdm and tensorboardx.
I also use logging to print information into log file
rather than print function.

__update May 17th, 2019:__ I tried the augmentation found by AutoAugment, while still using
origin model architecture. And I got 0.94 accuracy.

__update May 24th, 2019:__ I've got 0.96 accuracy on cifar10. I think it's enough. This repo then
will be used to keep track about my ideas, my fun of implementing interesting architectures.

## Environment

- Hardware: 2 GPUs
- Software: Pytorch, prefetch_generator, tensorboardX, tqdm

## Result

| Model                  | Test accuracy |
| -----------------------| ------------- |
| ResNet110              | 0.84          |
| ResNet110+Augmentation | 0.92          |
| ResNet110+AutoAugment  | 0.94          |
| WRN40+AutoAugment      | 0.96          |

I only tested ResNet with 110 layers. I will keep update for better
performance and for other architechture.

## TODO

- [ ] Improve test accuracy
- [ ] Test other architecture
