import keras
import sklearn
import pickle
from dotenv import load_dotenv
from os import environ

load_dotenv()


def test_libs():
    assert keras, 'keras missing'
    assert sklearn, 'sklearn missing'


def test_dataset():
    dataset_path = environ['dataset']

    assert dataset_path, 'dataset path not found'

    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    assert dataset, 'loading dataset failed.'


if __name__ == "__main__":
    test_libs()
    test_dataset()
