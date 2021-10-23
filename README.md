# AWS SageMaker

```
/opt
    ├── program
    |    ├─ train                        learning
    |    |
    │    ├─ predict.py                   predicate
    │    ├─ wsgi.py                      predicate
    │    ├─ nginx.conf                   predicate
    │    └─ serve                        predicate
    └──ml
        ├── input
        │   ├── config
        │   │   ├── hyperparameters.json This is a JSON file in which hyperparameters used for learning and other purposes are stored in dictionary format. Since all values are in string format, it is necessary to cast the values to the appropriate type when reading the file.
        │   └── data
        │       └── <channel_name>       It contains the input data for the corresponding channel. The data is copied from S3 before the learning process is executed.
        │           └── <input data>
        ├── model                        Outputs the model data resulting from the training.
        │   └── <model files>            The model data to be saved can be a single file or multiple files; SageMaker will automatically tar the model directory and store it in S3.
        └── output                       If a job fails during job execution, the program outputs an error message indicating the cause and other information.
            └── failure
```
