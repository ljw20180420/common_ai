# Introduction

This repository contains AI libraries commonly used for all my AI projects.

# Train

The `MyTrain` class train subclass of huggingface `PreTrainedModel`.

## `my_train_model`

The `my_train_model` method of `MyTrain` is the main method to train the model. The model can override `MyTrain`'s `my_train_model` method by implementing its own one.

The `my_train_model` has five parts.
1. Instantialize components include `my_generator`, `initializer`, `optimizer`, `lr_scheduler` and `metrics`.
2. Either call `my_initialize_model` to initialize model weights or load last epoch checkpoint to recover the states of model and other components in 1.
3. Train loop.
    - `my_train_epoch`.
    - `my_eval_epoch`.
    - Save epoch checkpoint.

### `my_initialize_model`

The model can override `MyTrain`'s `my_initialize_model` method by implementing its own one.

### Load epoch checkpoint

`MyTrain` call the model's `load_state_dict` method to load checkpoint. Generally, this will be the `load_state_dict` method of `torch.nn.module`. If the model override `torch.nn.module`'s `load_state_dict` method, then it should generally also override the `state_dict` method.

### `my_train_epoch`

The model can override `MyTrain`'s `my_train_epoch` method by implementing its own one.

### `my_eval_epoch`

The model can override `MyTrain`'s `my_eval_epoch` method by implementing its own one.

### Save epoch checkpoint

`MyTrain` call the model's `state_dict` method to save checkpoint.  Generally, this will be the `state_dict` method of `torch.nn.module`. If the model override `torch.nn.module`'s `state_dict` method, then it should generally also override the `load_state_dict` method.

# Test

The `MyTest` class test subclass of huggingface `PreTrainedModel`. `MyTest` will load the epoch saved by `MyTrain`.
1. If the model implement its own `my_train_model` method, then the model also needs to implement the `my_load_model` method for `MyTest` to load the model.
2. Otherwise, the `load_state_dict` method of the model (maybe overrided by the model) will be used by `MyTest` to load the model.
