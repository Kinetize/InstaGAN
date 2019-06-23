import os

import json
import numpy as np
import pickle


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
        key = img_name + '%s' % i
        hashtags_str = sample['hashtags']
        hashtags = []
        str_begin = False
        str_track = ""
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
    return tag_dict


def clean_up_tags_bin():
    """
    Idea: Remove all hash-tags which
    i) appear only n-times or less
    ii) do not contain ascii characters

    :return:
    """


def binary_representation(dict_tags, data_dir, embedding_filename, img_name='img'):
    """
    Counts all occurring hash-tags and assigns each hash-tag a one-hot vector.
    Then assigns each image a binary vector. Each entry represents if the hash-tag is assigned (1) or not(0).
    Saves the binary vector representations as a list into a .pickl file.

    :param dict: (String, [String]) Dictionary which points from the image name towards a list of hash-tags
    """

    # Make dictionaries which point towards the index of the given hashtag
    dict_tag_to_pos = {}
    dict_pos_to_tag = {}
    pos = 0
    for tag_list in dict_tags.values():
        for tag in tag_list:
            if tag not in dict_tag_to_pos:
                dict_tag_to_pos[tag] = pos
                dict_pos_to_tag[pos] = tag
                pos += 1
    binary_vec_size = pos

    # generate binary vectors
    list_bin_vec = []
    list_id = []
    for img_id in dict_tags:
        vec = np.zeros(binary_vec_size)
        # img_id = img_name + '%s' % i
        # tags = dict[img_id]
        for tag in dict_tags[img_id]:
            entry = dict_tag_to_pos[tag]
            vec[entry] = 1
        list_bin_vec.append(vec)
        list_id.append(img_id)
    save_embeddings = data_dir + embedding_filename
    save_filenames = data_dir + '/../data/filenames_bin.pickle'
    # print('Saving binary vectors at %s' % save_embeddings)
    with open(save_embeddings, 'wb') as f:
        pickle.dump(list_bin_vec, f)
    # print('Saving file-names at %s' % save_filenames)
    with open(save_filenames, 'wb') as f:
        pickle.dump(list_id, f)
    return list_bin_vec


def word2vec(dict_tags, data_dir, embedding_filename):
    """
    Utilizes a pre-trained word2vec embedding.
    (Maybe: Taking the mean or concatenate all embedding vectors as input)
    Saves the embedding for each hash-tag into a .pickle file

    :param dict_tags:
    :param model_dir:
    :param model_filename:
    :return:
    """

    words = []
    vectors = []
    with open(data_dir + embedding_filename, 'rb') as f:
        for l in f:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
    glove = {w: vectors[i] for i, w in enumerate(words)}

    # save word representations for each hash-tag which is in the data
    all_vec_rep = []
    list_id = []
    list_hash_tag = []
    for tags in dict_tags:
        image_vec_rep = []
        for tag in dict_tags[tags]:
            if tag in glove:
                image_vec_rep.append(glove[tag])
                list_hash_tag.append(tag)
        image_vec_rep = np.array(image_vec_rep)
        all_vec_rep.append(image_vec_rep)
        list_id.append(tags)
    save_embeddings = '../data/vec_emb.pickle'
    save_filenames = '../data/filenames_emb.pickle'
    save_img_hashtags = '../data/hashtags.txt'
    # print('Saving .pickle file at %s' % save_embeddings)
    with open(save_embeddings, 'wb') as f:
        pickle.dump(all_vec_rep, f)
    # print('Saving file-names at %s' % save_filenames)
    with open(save_filenames, 'wb') as f:
        pickle.dump(list_id, f)
    with open(save_img_hashtags, 'w') as f:
        for img_id in list_id:
            f.write(img_id + '\n')
            for tag in dict_tags[img_id]:
                f.write('\t' + tag + '\n')
#
#
# def load_embedding(data_dir, embedding_filename):
#     with open(data_dir + embedding_filename, 'rb') as f:
#         embeddings = pickle.load(f)
#         embeddings = np.array(embeddings)
#         # embedding_shape = [embeddings.shape[-1]]
#         print('embeddings: ', embeddings.shape)
#     return embeddings


if __name__ == '__main__':
    directory = os.path.dirname(os.path.realpath(__file__))
    save_file = '/../data/vec_bin.pickle'
    dict_img_to_tags = read_hashtags('output')

    bin_vec = binary_representation(dict_img_to_tags, directory, save_file)

    model_dir = os.path.dirname(os.path.realpath(__file__)) + '/word2vec_glove'
    model_filename = '/glove.twitter.27B.25d.txt'
    word2vec(dict_img_to_tags, model_dir, model_filename)
    print('done.')
