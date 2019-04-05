# Experiment setup

To conduct a proper experiment, there are several options that are

### Classification type
```
clf_type = 'nn'  # lr, nn
```
By setting this option, you can use either Logistic Regression or Neural Neural as a classificator. 

### Aggregation strategy
```
aggregation_options = 'max'  # avg, max
```
As we're conducting distant supervision over split_samples, this sets up the default strategy for aggregating (combing) samples back to channels.

### Split strategy
```
split_options = {
    'type': 'video',  # video, episodes
    'mean': False,  # True, False
    'config': 'IS09_emotion',  # IS09_emotion, IS12_speaker_trait
    'speech_embeddings': {
        'mean': False  # True, False
    }
}
```
Same as for aggregating we need a default strategy to split channels, so there are following options:
* __type__ - split channel either by speech episodes (episodes) or videos (video)
* mean - if _type_ is video, then you can decide if mean should be used (from all episoedes) True or use only the first speech episode (False)
* config - select which configuration should be used for Open Smile embeddings
* speech_embeddings.mean - Same as _mean_
```

### NN architecture
nn_arch = {
    'layers': [
        ('dropout', 0.2),
        ('dense', 128, 'relu'),
        ('dropout', 0.2),
        ('dense', 64, 'tanh'),
        ('dropout', 0.20),
        ('dense', 3, 'softmax'),
    ],
    'optimizer': 'adagrad',
    'batch_size': 75,
    'epochs': 35
}
```

Shorthand of writing keras model, for layers currently supporting Dense, Dropout. To check how this option is being used check [neural_network](https://github.com/yoandinkov/interspeech/blob/master/neural_network.py) file.

```
transformation_options = {
    'bert_fulltext': 1,  # 0,1
    'numerical': 1,  # 0,1
    'nela_desc': 1,  # 0,1
    'bert_subs': 1,  # 0,1
    'open_smile': 1,  # 0,1
    'speech_embeddings': 0,  # 0,1
}
```

# Single vs batch execution
To actually run experiments you will need to execute either `single_experiment.py` or `batch_execution.py`.