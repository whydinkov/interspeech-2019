## Motivation
Current repository is used as an experiment evaluation system. Generated results are being used in a research inspired by following conference https://www.interspeech2019.org/.

## Dataset
To run code locally, you'll need a specific dataset as a starting point which can be found at [Кaggle](https://www.kaggle.com/yoandinkov/youtubepoliticalbias). More information about dataset can be found in repository's [docs files](docs/dataset.md). There is an additional step of transforming dataset to a pickle instance which is explained [here](docs/environment_setup.md#environment-variables)

## Prerequisites
* Setup `python 3.7+`, `conda` on Ubuntu 18.04 LTS
* Install environment via `conda env create yoandinkov/interspeech` or check it [here](https://anaconda.org/yoandinkov/interspeech)
* Set up `.env` file, information can be found [here](docs/environment_setup.md)
* Open `tests/` and run `python prerequisites.py` if no error messages are displayed, you are good to go

## Run experiments
* Open `src/` and run `python main.py` - there are default values so by only doing this code will be executed, however it's highly recommended to read [the experiment setup documentation](docs/experiment_setup.md) first.

## Feedback
* Any feedback is appreatied. For specific questions/code reviews new issues/PRs can be opened/created.

## Acknowledgements
This research is part of the [Tanbih project](http://tanbih.qcri.org/), which aims to limit the effect of "fake news", propaganda and media bias by making users aware of what they are reading. The project is developed in collaboration between the Qatar Computing Research Institute (QCRI), HBKU and the MIT Computer Science and Artificial Intelligence Laboratory (CSAIL).

This research is also partially supported by Project UNITe BG05M2OP001-1.001-0004 funded by the OP "Science and Education for Smart Growth" and the EU via the ESI Funds.