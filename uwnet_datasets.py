"""Contains the standard train/test splits for the cyclegan data.
Author: Honey Gupta (hn.gpt1@gmail.com)
"""

"""The size of each dataset. Usually it is the maximum number of images from
each domain."""
DATASET_TO_SIZES = {
    'air2water_train': 1500,
    'hazelines': 58
}

"""The path to the output csv file."""
PATH_TO_CSV = {
     'air2water_train' : 'input/air2water_train.csv',
     'hazelines' : 'input/hazelines.csv'
}

"""The image types of each dataset. Currently only supports .jpg or .png"""
DATASET_TO_IMAGETYPE = {
    'air2water_train':	'.jpg',
    'hazelines': '.jpg'
}


