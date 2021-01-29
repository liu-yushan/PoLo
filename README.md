# Neural Multi-Hop Reasoning With Logical Rules on Biomedical Knowledge Graphs

Tensorflow implementation of the method MINERVA+.


<h2> Credits</h2>

This implementation is based on [Shehzaad Dhuliawala's repository](https://github.com/shehzaadzd/MINERVA), which contains the code for the paper [Go for a Walk and Arrive at the Answer - Reasoning over Paths in Knowledge Bases using Reinforcement Learning](https://arxiv.org/abs/1711.05851).

<h2> How To Run </h2>

The dependencies are specified in [requirements.txt](requirements.txt). To run MINERVA+, use one of the config files or create your own. For an explanation of each hyperparameter, refer to the [README file in the configs folder](configs/README.md).

Then, run the command
```
./run.sh configs/${config_file}.sh
```


<h2> Data Format </h2>

<h5> Triple format </h5>

KG triples need to be written in the format ```subject predicate object```, with tabs as separators. Furthermore, MINERVA+ uses inverse relations, so it is important to add the inverse triple for each fact in the KG. The prefix  ```_``` is used before a predicate to signal the inverse relation, e.g., the inverse triple for ```Germany hasCapital Berlin``` is ```Berlin _hasCapital Germany```.

<h5> File format </h5>

Datasets should have the following files:
```
dataset
    ├── graph.txt
    ├── train.txt
    ├── dev.txt
    └── test.txt
```

Where:

```train.txt``` contains all train triples.

```dev.txt``` contains all validation triples.

```test.txt``` contains all test triples.

```graph.txt``` contains all triples of the KG except for ```dev.txt```, ```test.txt```, the inverses of ```dev.txt```, and the inverses of ```test.txt```.

Finally, two vocab files are needed, one for the entities and one for the relations. These can be created by using the [```create_vocab.py``` file](mycode/data/preprocessing_scripts/create_vocab.py).
