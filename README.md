# Introduction

This repository contains AI libraries commonly used for all my AI projects.

# Train

The `MyTrain` class train subclass of huggingface `PreTrainedModel`. `model.state_dict` and `model.load_state_dict` must be consistent.

```mermaid
---
title: MyTrain.__call__
---
flowchart TD
    INST[instantiate model] --> MODE{{evaluation only?}}
    MODE -- yes --> EVALMODEL[<code>MyTrain.my_eval_model</code>]

    subgraph EVALMODEL[<code>MyTrain.my_eval_model</code>]
        EVALINSTCOMPS[instantiate components] --> EVALINSTMETRICS[instantiate metrics] --> EVALLOOP[eval loop]
    end

    subgraph EVALLOOP[eval loop]
        CHECKCONSISTENCY[check config consistency] --> EVALLOADCHECKPOINT[load checkpoint] --> EVALEPOCHBRANCH{{implement <code>model.my_eval_epoch</code>?}}
        EVALEPOCHBRANCH -- yes --> CUSTOMEVAL[<code>model.my_eval_epoch</code>]
        EVALEPOCHBRANCH -- no --> COMMONEVAL[<code>MyTrain.my_eval_epoch</code>]
        CUSTOMEVAL --> UPDATECONFIGPERFORM[update configuration and performance]
        COMMONEVAL --> UPDATECONFIGPERFORM[update configuration and performance]
    end

    MODE -- no --> COMMONTRAIN[<code>MyTrain.my_train_model</code>]

    subgraph COMMONTRAIN[<code>MyTrain.my_train_model</code>]
        TRAININSTCOMPS[instantiate components] --> TRAININSTMETRICS[instantiate metrics] --> CONTINUETRAIN{{last epoch is -1?}}
        CONTINUETRAIN -- yes --> INITWEIGHT[initialize model weights]
        CONTINUETRAIN -- no --> TRAINLOADCHECK[load checkpoint]
        INITWEIGHT --> TRAINLOOP[train loop]
        TRAINLOADCHECK --> TRAINLOOP
    end
    subgraph TRAINLOOP[train loop]
        M{{implement <code>model.my_train_epoch</code>?}}
        M -- yes --> N[<code>model.my_train_epoch</code>]
        M -- no --> O[<code>MyTrain.my_train_epoch</code>]
        N --> P{{implement <code>model.my_eval_epoch</code>?}}
        O --> P
        P -- yes --> Q[<code>model.my_eval_epoch</code>]
        P -- no --> R[<code>MyTrain.my_eval_epoch</code>]
        Q --> UPDATELR[update learning rate]
        R --> UPDATELR
        UPDATELR --> S[save current epoch and configuration] --> T[save performance] --> U[save checkpoint] --> EARLYSTOP[check early stopping]
    end
```

# Test

The `MyTest` class test subclass of huggingface `PreTrainedModel`. `MyTest` will load the epoch saved by `MyTrain`. If `model.my_train_model` is implemented, then the corresponding `model.my_load_model` is necessary.

```mermaid
---
title: MyTest.__call__
---
flowchart TD
    INSTMODEL[instantiate model] --> INSTMETRIC[instantiate metrics] --> LOADCHECK[load checkpoint] --> F[calculate test metrics] --> G[save test metrics]
```

# Metric

The metric classes should implement three methods.
1. `__init__` intialized the parameters and metric state.
2. `step` process the batchs. It receives:
    - `df`: the data frame returned by the model's `eval_output` method.
    - `examples`: the examples in the dataset.
    - `batch`: the batch returned by the model's `data_collator`.
3. `epoch` accumulate all batch results, reinitialize the metric state and return the final metric.
