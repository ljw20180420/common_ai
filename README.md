# Introduction

This repository contains AI libraries commonly used for all my AI projects.

# Train

The `MyTrain` class train subclass of huggingface `PreTrainedModel`. `model.state_dict` and `model.load_state_dict` must be consistent.

```mermaid
---
title: MyTrain.__call__
---
flowchart TD
    INST[instantiate model and random generator] --> TRAININSTMETRICS[instantiate metrics] --> MODE{{evaluation only?}}
    MODE -- yes --> EVALMODEL[<code>MyTrain.my_eval_model</code>]

    subgraph EVALMODEL[<code>MyTrain.my_eval_model</code>]
        direction TB
        EVALLOOP[eval loop]
    end

    subgraph EVALLOOP[eval loop]
        direction TB
        CHECKCONSISTENCY[check config consistency] --> EVALLOADCHECKPOINT[load checkpoint for model and generator] --> EVALDEVICE[set model device] --> EVALDATALOADER[setup data loader] --> EVALEPOCHBRANCH{{implement <code>model.my_eval_epoch</code>?}}
        EVALEPOCHBRANCH -- yes --> CUSTOMEVAL[<code>model.my_eval_epoch</code>]
        EVALEPOCHBRANCH -- no --> COMMONEVAL[<code>MyTrain.my_eval_epoch</code>]
        CUSTOMEVAL --> UPDATECONFIGPERFORM[update configuration]
        COMMONEVAL --> UPDATECONFIGPERFORM
    end

    MODE -- no --> COMMONTRAIN[<code>MyTrain.my_train_model</code>]

    subgraph COMMONTRAIN[<code>MyTrain.my_train_model</code>]
        direction TB
        CONTINUETRAIN{{last epoch is -1?}}
        CONTINUETRAIN -- yes --> HASINIT{{implement <code>model.my_initialize_model</code>?}} -- yes --> CUSTOMINIT[<code>model.my_initialize_model</code>?]
        HASINIT -- no --> INITWEIGHT[initialize model weights by <code>my_initializer</code>]
        CONTINUETRAIN -- no --> TRAINLOADCHECK[load checkpoint for model and random generator]
        CUSTOMINIT --> TRAINDEVICE[set model device]
        INITWEIGHT --> TRAINDEVICE
        TRAINLOADCHECK --> TRAINDEVICE
        TRAINDEVICE --> INSTOPSC[instantiate optimizer and lr scheduler] --> CONTINUETRAIN2{{last epoch is -1?}} -- no --> TRAINCHECKOPSC[load checkpoint for optimizer and lr scheduler] --> SETUPOPSC[setup optimizer and lr_scheduler]
        CONTINUETRAIN2{{last epoch is -1?}} -- yes --> SETUPOPSC
        SETUPOPSC --> TRAINDATALOADER[setup data loader] --> INSTEARLYSTOP[instantiate early stopping] --> TRAINLOOP[train loop]
    end

    subgraph TRAINLOOP[train loop]
        direction TB
        M{{implement <code>model.my_train_epoch</code>?}}
        M -- yes --> N[<code>model.my_train_epoch</code>]
        M -- no --> O[<code>MyTrain.my_train_epoch</code>]
        N --> P{{implement <code>model.my_eval_epoch</code>?}}
        O --> P
        P -- yes --> Q[<code>model.my_eval_epoch</code>]
        P -- no --> R[<code>MyTrain.my_eval_epoch</code>]
        Q --> UPDATELR[update learning rate]
        R --> UPDATELR
        UPDATELR --> TRAINSAVE[save epoch configuration and checkpoint] --> EARLYSTOP[check early stopping]
    end
```

# Test

The `MyTest` class test subclass of huggingface `PreTrainedModel`. `MyTest` will load the epoch saved by `MyTrain`. If `model.my_train_model` is implemented, then the corresponding `model.my_load_model` is necessary.

```mermaid
---
title: MyTest.__call__
---
flowchart TD
    INSTMODEL[instantiate model and random generator] --> INSTMETRIC[instantiate metrics] --> LOADCHECK[load checkpoint for model and random generator] --> TESTDEVICE[set model device] --> TESTDATA[setup data loader] --> TESTMODEL[test model] --> TESTSAVE[save metrics]
```

# Metric

The metric classes should implement three methods.
1. `__init__` intialized the parameters and metric state.
2. `step` process the batchs. It receives:
    - `df`: the data frame returned by the model's `eval_output` method.
    - `examples`: the examples in the dataset.
    - `batch`: the batch returned by the model's `data_collator`.
3. `epoch` accumulate all batch results, reinitialize the metric state and return the final metric.
