import os

import json
import numpy as np
import pickle


def read_hashtags(filename, img_name='img'):
    """
    Reads the hash-tags from crawler output and converts them into a dictionary
    :param filename:
    :param img_name:
    :return:
    """
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    file_path = '%s/crawler/' % curr_dir + filename
    with open(file_path, 'r') as file:
        data = json.load(file)
    img_num = len(data)
    print('Number of imgages:\t', img_num)
    tag_dict = {}
    for i, sample in enumerate(data):
        key = img_name + '%s' % i
        hashtags_str = sample['hashtags']
        hashtags = []
        str_begin = False
        str_track = ""
        # print('Hashtags:\t', hashtags_str)
        for char in hashtags_str:
            if str_begin:
                if char == "\'":
                    str_begin = False
                    hashtags.append(str_track)
                    str_track = ""
                else:
                    str_track += char
            if char == '#':
                str_begin = True
        tag_dict[key] = hashtags
        # print(key, '\t', tag_dict[key])
    return tag_dict


def clean_up_tags_bin():
    """
    Idea: Remove all hash-tags which
    i) appear only n-times or less
    ii) do not contain ascii characters

    :return:
    """


def binary_representation(dict, data_dir, embedding_filename, img_name='img'):
    """
    Counts all occurring hash-tags and assigns each hash-tag a one-hot vector.
    Then assigns each image a binary vector. Each entry represents if the hash-tag is assigned (1) or not(0).
    Saves the binary vector representations as a list into a .pickl file.

    :param dict: (String, [String]) Dictionary which points from the image name towards a list of hash-tags
    """
    # Make dictionaries which point towards the index of the given hashtag
    img_number = len(dict)
    dict_tag_to_pos = {}
    dict_pos_to_tag = {}
    pos = 0
    for tag_list in dict.values():
        for tag in tag_list:
            if tag not in dict_tag_to_pos:
                dict_tag_to_pos[tag] = pos
                dict_pos_to_tag[pos] = tag
                pos += 1
    binary_vec_size = pos
    print(binary_vec_size)
    # generate binary vectors
    list_bin_vec = []
    for i in range(img_number):
        vec = np.zeros(binary_vec_size)
        tags = dict[img_name + '%s' % i]
        for tag in tags:
            entry = dict_tag_to_pos[tag]
            vec[entry] = 1
        list_bin_vec.append(vec)
    save_loc = data_dir + embedding_filename
    print('Saving .pickle file at %s' % save_loc)
    with open(save_loc, 'wb') as f:
        pickle.dump(list_bin_vec, f)
    return list_bin_vec


def word2vec(dict):
    """
    Utilizes a pre-trained word2vec embedding.
    (Maybe: Taking the mean or concatenate all embedding vectors as input)
    Saves the embedding for each hash-tag into a .pickle file

    :param dict:
    :return:
    """


def load_embedding(data_dir, embedding_filename):
    with open(data_dir + embedding_filename, 'rb') as f:
        embeddings = pickle.load(f)
        embeddings = np.array(embeddings)
        # embedding_shape = [embeddings.shape[-1]]
        print('embeddings: ', embeddings.shape)
    return embeddings


if __name__ == '__main__':
    directory = os.path.dirname(os.path.realpath(__file__))
    save_file = '/tryout.pickle'
    dict_img_to_tags = read_hashtags('output')
    # binary_representation(dict)
    bin_vec = binary_representation(dict_img_to_tags, directory, save_file)
    embeddings = load_embedding(directory, save_file)
    for tags in dict_img_to_tags:
        print(tags, '\t', dict_img_to_tags[tags])