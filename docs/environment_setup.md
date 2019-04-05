## General information
All tests have been run on a Ubuntu 18.04 LTS with Tesla P100 16GB. Language is Python 3.

## Environment setup
For environment isolation [miniconda-3](https://docs.conda.io/projects/conda/en/latest/) was the tool of choice. After downloading and installing conda it is advisable to use developers' environment, which is shared [here](https://anaconda.org/yoandinkov/interspeech)

## Environment variables
This project uses [dotenv](https://github.com/theskumar/python-dotenv), so you need to have a local .env file in root directory with following properties to run run the code locally:

```
dataset='{path to dataset*}'
experiments_output='{path to experiment output folder (will be created automatically)}'
```

* Dataset is expected to be in pickle format, so if you're downloading it first from kaggle, please save it to pickle format first and then refer it in .env file.