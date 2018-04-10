
# coding: utf-8

# ![DISI](../resources/DISI.jpeg)

# ### Import

# In[1]:


import json
import subprocess
import numpy as np


# ### Load JSON configuration file

# In[2]:


# set the configuration file
CONFIGURATION_FILE = "k_fold_config.json"

# open the configuration file
json_configuration_file = open(CONFIGURATION_FILE, "r", encoding="utf8")

# load the configuration file
CONFIGURATION = json.load(json_configuration_file)


# In[3]:


OUTPUT_CONFIGURATION_FILE_NAME = "config.json"


# In[4]:


K = int(CONFIGURATION["k"])

OUTPUT_DIR = CONFIGURATION["output_dir"]
if len(OUTPUT_DIR) == 0:
    OUTPUT_DIR += "{0}_fold_cross_validation".format(K)
else:
    OUTPUT_DIR += "/{0}_fold_cross_validation".format(K)

TRAIN_FILE = CONFIGURATION["train_file"]
TRAIN_FEATS_FILE = CONFIGURATION["train_feats_file"]

TRAIN_TEMP_FILE = "{0}/temp_train.txt".format(OUTPUT_DIR)
TRAIN_TEMP_FEATS_FILE = "{0}/temp_train_feats.txt".format(OUTPUT_DIR)
TEST_TEMP_FILE = "{0}/temp_test.txt".format(OUTPUT_DIR)
TEST_TEMP_FEATS_FILE = "{0}/temp_test_feats.txt".format(OUTPUT_DIR)

HANDLE_UNK = CONFIGURATION["handle_unk"]
NGRAM_SIZE = CONFIGURATION["ngram_size"]
SMOOTHING = CONFIGURATION["smoothing"]
ADDITIONAL_FEATURE = CONFIGURATION["additional_feature"]
IMPROVEMENT = CONFIGURATION["improvement"]

BACKOFF = CONFIGURATION["backoff"]
BINS = CONFIGURATION["bins"]
WITTEN_BELL_K = CONFIGURATION["witten_bell_k"]
DISCOUNT_D = CONFIGURATION["discount_D"]


# In[5]:


assert(subprocess.call("[[ -e config.json ]] && cp config.json config_bak.json", shell = True) == 0),"> unable to make a copy of configuration file"


# In[6]:


# remove the output directory if already exists
assert(subprocess.call("rm -rf {0}".format(OUTPUT_DIR), shell = True) == 0),"> unable to remove {0}".format(OUTPUT_DIR)

# make another directory with the specified output directory name
assert(subprocess.call("mkdir -p {0}".format(OUTPUT_DIR), shell = True) == 0),"> unable to mkdir {0}".format(OUTPUT_DIR)


# In[7]:


def read_sentences(input_file_path):
    input_file = open(input_file_path, "r", encoding="utf8")
    
    input_file_sentences = []
    sentence = []
    
    for line in input_file:
        line = line.replace("\n", "")
        
        if len(line) == 0:
            sentence.append("\n")
            input_file_sentences.append(sentence)
            sentence = []
            continue
        
        sentence.append(line)
        
    input_file.close()
    
    return input_file_sentences


# In[8]:


train_sentences = read_sentences(TRAIN_FILE)
train_feats_sentences = read_sentences(TRAIN_FEATS_FILE)

assert(len(train_sentences) == len(train_feats_sentences))


# In[9]:


train_folds = np.array_split(train_sentences, K)
train_feats_folds = np.array_split(train_feats_sentences, K)


# In[10]:


def write_fold_to_file(fold, output_file_path):
    output_file = open(output_file_path, "w", encoding="utf8")
    for sentence in fold:
        for line in sentence:
            line = line.replace("\n", "")
            output_file.write("{0}\n".format(line))
    output_file.close()


# In[11]:


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

        configuration_file = open(OUTPUT_CONFIGURATION_FILE_NAME, 'w', encoding="utf8")
        json.dump(configuration, configuration_file)
        configuration_file.close()

        assert(subprocess.call("python3 slu.py", shell = True) == 0),                        "> unable to run the script"
        print("\n")
except:
    pass


# In[12]:


assert(subprocess.call("[[ -e config_bak.json ]] && mv config_bak.json config.json", shell = True) == 0),"> unable to make a copy of configuration file"

