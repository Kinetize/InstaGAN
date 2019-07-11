import os

import json
import numpy as np
import pickle
import re
import pprint


def read_hashtags(filename, img_name='img'):
    """
    Reads the hash-tags from crawler output and converts them into a dictionary.
    :param filename:
    :param img_name:
    :return:
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = '%s/../crawler/' % curr_dir + filename
    with open(file_path, 'r') as file:
        data = json.load(file)
    img_num = len(data)
    print('Number of images:\t', img_num)
    tag_dict = {}
    for i, sample in enumerate(data):
        key_regex = re.compile("([^\/]+)\/$")
        key = key_regex.findall(sample['key'])[0]
        hashtags = [s.replace("#", "") for s in sample['hashtags']]
        tag_dict[key] = hashtags
    return tag_dict

if __name__ == '__main__':
    hashtag_dict = read_hashtags('insta_output')
    counts = {}

    for key in hashtag_dict.keys():
        for hashtag in hashtag_dict[key]:
            try:
                counts[hashtag] += 1
            except KeyError:
                counts[hashtag] = 1

    printer = pprint.PrettyPrinter(indent=2)
    printer.pprint(sorted(counts.items(), key=lambda item: item[1])[-100:])
