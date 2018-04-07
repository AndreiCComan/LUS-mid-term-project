
# coding: utf-8

# ![DISI](../resources/DISI.jpeg)

# ### Import

# In[ ]:


import json
import subprocess
import numpy as np


# In[ ]:


K = 10


# In[ ]:


CONFIGURATION_FILE_NAME = "config.json"


# In[ ]:


OUTPUT_DIR = "{0}_fold_cross_validation".format(K)
TRAIN_FILE = "NLSPARQL.train.data"
TRAIN_FEATS_FILE = "NLSPARQL.train.feats.txt"

TRAIN_TEMP_FILE = "{0}/temp_train.txt".format(OUTPUT_DIR)
TRAIN_TEMP_FEATS_FILE = "{0}/temp_train_feats.txt".format(OUTPUT_DIR)
TEST_TEMP_FILE = "{0}/temp_test.txt".format(OUTPUT_DIR)
TEST_TEMP_FEATS_FILE = "{0}/temp_test_feats.txt".format(OUTPUT_DIR)

HANDLE_UNK = "uniform"
NGRAM_SIZE = 4
SMOOTHING = "kneser_ney"
ADDITIONAL_FEATURE = "none"
IMPROVEMENT = "naive"

BACKOFF = "false"
BINS = -1
WITTEN_BELL_K = 1.0
DISCOUNT_D = -1


# In[ ]:


assert(subprocess.call("[[ -e config.json ]] && cp config.json config_bak.json", shell = True) == 0),"> unable to make a copy of configuration file"


# In[ ]:


# remove the output directory if already exists
assert(subprocess.call("rm -rf {0}".format(OUTPUT_DIR), shell = True) == 0),"> unable to remove {0}".format(OUTPUT_DIR)

# make another directory with the specified output directory name
assert(subprocess.call("mkdir -p {0}".format(OUTPUT_DIR), shell = True) == 0),"> unable to mkdir {0}".format(OUTPUT_DIR)


# In[ ]:


def read_sentences(input_file_path):
    input_file = open(input_file_path, "r", encoding="utf8")
    
    input_file_sentences = []
    sentence = []
    
    for line in input_file:
        line = line.replace("\n", "")
        
        if len(line) == 0:
            input_file_sentences.append(sentence)
            sentence = []
        
        sentence.append(line)
        
    input_file.close()
    
    return input_file_sentences


# In[ ]:


train_sentences = read_sentences(TRAIN_FILE)
train_feats_sentences = read_sentences(TRAIN_FEATS_FILE)

assert(len(train_sentences) == len(train_feats_sentences))


# In[ ]:


train_folds = np.array_split(train_sentences, K)
train_feats_folds = np.array_split(train_feats_sentences, K)


# In[ ]:


def write_fold_to_file(fold, output_file_path):
    output_file = open(output_file_path, "w", encoding="utf8")
    for sentence in fold:
        for line in sentence:
            output_file.write("{0}\n".format(line))
    output_file.close()


# In[ ]:


try:
    for fold_index in range(len(train_folds)):

        print("> fold index:\t{0}\n".format(fold_index))

        test_fold = train_folds[fold_index]
        test_feats_fold = train_feats_folds[fold_index]

        write_fold_to_file(test_fold, TEST_TEMP_FILE)
        write_fold_to_file(test_feats_fold, TEST_TEMP_FEATS_FILE)

        train_folds_indices = list(range(len(train_folds)))
        train_folds_indices.remove(fold_index)

        train = []
        train_feats = []

        for train_fold_index in train_folds_indices:
            train.append(train_folds[train_fold_index])
            train_feats.append(train_feats_folds[train_fold_index])

        train = np.concatenate(train)
        train_feats = np.concatenate(train_feats)

        write_fold_to_file(train, TRAIN_TEMP_FILE)
        write_fold_to_file(train_feats, TRAIN_TEMP_FEATS_FILE)

        configuration = {}
        configuration["output_dir"] = "{0}/fold_{1}".format(OUTPUT_DIR, fold_index)
        configuration["train_file"] = TRAIN_TEMP_FILE
        configuration["train_feats_file"] = TRAIN_TEMP_FEATS_FILE
        configuration["test_file"] = TEST_TEMP_FILE
        configuration["test_feats_file"] = TEST_TEMP_FEATS_FILE
        configuration["handle_unk"] = HANDLE_UNK
        configuration["ngram_size"] = NGRAM_SIZE
        configuration["smoothing"] = SMOOTHING
        configuration["additional_feature"] = ADDITIONAL_FEATURE
        configuration["improvement"] = IMPROVEMENT
        configuration["backoff"] = BACKOFF
        configuration["bins"] = BINS
        configuration["witten_bell_k"] = WITTEN_BELL_K
        configuration["discount_D"] = DISCOUNT_D

        configuration_file = open(CONFIGURATION_FILE_NAME, 'w', encoding="utf8")
        json.dump(configuration, configuration_file)
        configuration_file.close()

        assert(subprocess.call("python3 slu.py", shell = True) == 0),                        "> unable to run the script"
        print("\n")
except:
    pass


# In[ ]:


assert(subprocess.call("[[ -e config_bak.json ]] && mv config_bak.json config.json", shell = True) == 0),"> unable to make a copy of configuration file"

