# Introduction

This repository contains AI libraries commonly used for all my AI projects.

# Train

The `MyTrain` class train subclass of huggingface `PreTrainedModel`. `model.state_dict` and `model.load_state_dict` must be consistent.

```mermaid
---
title: MyTrain.__call__
---
flowchart TD
    A[initialize model] --> B{{implement <code>model.my_train_model</code>?}}
    B -- Yes --> C[<code>model.my_train_model</code>]
    B -- No --> D[<code>MyTrain.my_train_model</code>]

    subgraph D[<code>MyTrain.my_train_model</code>]
        E["`
            initialize:
                - my_generator
                - initializer
                - optimizer
                - lr_scheduler
                - metrics
        `"] --> F{{last epoch is -1?}}
        F -- yes --> G{{implement <code>model.my_initialize_model</code>?}}
        G -- yes --> H[<code>model.my_initialize_model</code>]
        G -- no --> I[<code>MyTrain.my_initialize_model</code>]
        F -- no --> J["`
            load:
                - my_generator
                - optimizer
                - lr_scheduler
        `"] --> K[<code>model.load_state_dict</code>]
        H --> L[train loop]
        I --> L
        K --> L
    end
    subgraph L[train loop]
        M{{implement <code>model.my_train_epoch</code>?}}
        M -- yes --> N[<code>model.my_train_epoch</code>]
        M -- no --> O[<code>MyTrain.my_train_epoch</code>]
        N --> P{{implement <code>model.my_eval_epoch</code>?}}
        O --> P
        P -- yes --> Q[<code>model.my_eval_epoch</code>]
        P -- yes --> R[<code>MyTrain.my_eval_epoch</code>]
        Q --> S[save current epoch and configuration]
        R --> S
        S --> T[save performance] --> U["`
            save:
                - my_generator
                - optimizer
                - lr_scheduler
        `"] --> V[<code>model.state_dict</code>]
    end
```

# Test

The `MyTest` class test subclass of huggingface `PreTrainedModel`. `MyTest` will load the epoch saved by `MyTrain`. If `model.my_train_model` is implemented, then the corresponding `model.my_load_model` is necessary.

```mermaid
---
title: MyTest.__call__
---
flowchart TD
    A[initialize metric] --> B[initialize model] --> C{{implement <code>model.my_load_model</code>?}}
    C -- yes --> D[<code>model.my_load_model</code>]
    C -- no --> E[<code>model.load_state_dict</code>]
    D --> F[calculate test metrics]
    E --> F
    F --> G[save test metrics]
```

# Metric

The metric classes should implement three methods.
1. `__init__` intialized the parameters and metric state.
2. `step` process the batchs. It receives:
    - `df`: the data frame returned by the model's `eval_output` method.
    - `examples`: the examples in the dataset.
    - `batch`: the batch returned by the model's `data_collator`.
3. `epoch` accumulate all batch results, reinitialize the metric state and return the final metric.
