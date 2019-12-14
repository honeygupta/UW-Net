"""Create datasets for training and testing.
Modified by: Honey Gupta (hn.gpt1@gmail.com)
"""

import os
import csv
import click
import random

import uwnet_datasets

def create_list(foldername, fulldir=True, suffix=".jpg"):
    """

    :param foldername: The full path of the folder.
    :param fulldir: Whether to return the full path or not.
    :param suffix: Filter by suffix.

    :return: The list of filenames in the folder with given suffix.

    """
    file_list_tmp = os.listdir(foldername)
    file_list_tmp.sort()
    file_list = []
    if fulldir:
        for item in file_list_tmp:
            file_list.append(os.path.join(foldername, item))
    else:
        for item in file_list_tmp:
            file_list.append(item)
    return file_list

@click.command()
@click.option('--image_path_a',
              type=click.STRING,
              default='/home/honey/honey/underwater/nyu/resized_hazy_smoothDepth',
              help='The path to folder containing the .npy RGBD images.')
@click.option('--image_path_b',
              type=click.STRING,
              default='/home/honey/honey/underwater/datasets/Berman_hazelines/final_rgb',
              help='The path to the folder containing the underwater images.')
@click.option('--dataset_name',
              type=click.STRING,
              default='hazelines',
              help='The name of the dataset in uwnet_dataset.')
@click.option('--do_shuffle',
              type=click.BOOL,
              default=False,
              help='Whether to shuffle images when creating the dataset.')
@click.option('--mode',
              type=click.STRING,
              default='test',
              help='Choose one among ["train","test"].')

def create_dataset(image_path_a, image_path_b,
                   dataset_name, do_shuffle, mode):
    if mode == 'train':
        list_a = create_list(image_path_a, True,
                         uwnet_datasets.DATASET_TO_IMAGETYPE[dataset_name])

    list_b = create_list(image_path_b, True,
                        uwnet_datasets.DATASET_TO_IMAGETYPE[dataset_name])

    output_path = uwnet_datasets.PATH_TO_CSV[dataset_name]

    num_rows = uwnet_datasets.DATASET_TO_SIZES[dataset_name]
    all_data_tuples = []
    
    if mode == 'train':
        for i in range(num_rows):
            all_data_tuples.append((
            list_a[i % len(list_a)],
            list_b[i % len(list_b)]
            ))
    elif mode == 'test':
        all_data_tuples = list_b

    if do_shuffle is True:
        random.shuffle(all_data_tuples)
    
    with open(output_path, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        if mode == 'train':
            for data_tuple in enumerate(all_data_tuples):
                csv_writer.writerow(list(data_tuple[1]))
        elif mode == 'test':
            for data_tuple in all_data_tuples:
                csv_writer.writerow((data_tuple,))

if __name__ == '__main__':
    create_dataset()
