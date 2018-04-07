
# coding: utf-8

# ![DISI](../resources/DISI.jpeg)

# ### Import

# In[ ]:


import json
import subprocess
import pprint


# In[ ]:


CONFIGURATION_FILE_NAME = "config.json"


# In[ ]:


OUTPUT_DIR = "parameters_search"
TRAIN_FILE = "NLSPARQL.train.data"
TRAIN_FEATS_FILE = "NLSPARQL.train.feats.txt"
TEST_FILE = "NLSPARQL.test.data"
TEST_FEATS_FILE = "NLSPARQL.test.feats.txt"

HANDLE_UNK = ["uniform", "cut_off_1", "cut_off_2"]
NGRAM_SIZE = [2, 3, 4, 5, 6]
SMOOTHINGS = ["witten_bell", "kneser_ney", "katz", "absolute", "presmoothed"]
ADDITIONAL_FEATURES = ["none", "lemma", "lemmapos", "tokenpos"]
IMPROVEMENTS = ["none", "wise", "naive"]

BACKOFF = "false"
BINS = -1
WITTEN_BELL_K = 1.0
DISCOUNT_D = -1


# In[ ]:


assert(subprocess.call("[[ -e config.json ]] && cp config.json config_bak.json", shell = True) == 0),"> unable to make a copy of configuration file"


# In[ ]:


try:
    for handle_unk in HANDLE_UNK:
        for ngram_size in NGRAM_SIZE:
            for smoothing_algorithm in SMOOTHINGS:
                for additional_feature in ADDITIONAL_FEATURES:
                    for improvement in IMPROVEMENTS:
                        configuration = {}
                        configuration["output_dir"] = OUTPUT_DIR
                        configuration["train_file"] = TRAIN_FILE
                        configuration["train_feats_file"] = TRAIN_FEATS_FILE
                        configuration["test_file"] = TEST_FILE
                        configuration["test_feats_file"] = TEST_FEATS_FILE
                        configuration["handle_unk"] = handle_unk
                        configuration["ngram_size"] = ngram_size
                        configuration["smoothing"] = smoothing_algorithm
                        configuration["additional_feature"] = additional_feature
                        configuration["improvement"] = improvement
                        configuration["backoff"] = BACKOFF
                        configuration["bins"] = BINS
                        configuration["witten_bell_k"] = WITTEN_BELL_K
                        configuration["discount_D"] = DISCOUNT_D

                        configuration_file = open(CONFIGURATION_FILE_NAME, 'w', encoding="utf8")
                        json.dump(configuration, configuration_file)
                        configuration_file.close()

                        print("+---------------+")
                        print("| CONFIGURATION |")
                        print("+---------------+----------------------------------------------")
                        pp = pprint.PrettyPrinter()
                        pp.pprint(configuration)

                        assert(subprocess.call("python3 slu.py", shell = True) == 0),                        "> unable to run the script"

                        print("\n")
except:
    pass


# In[ ]:


assert(subprocess.call("[[ -e config_bak.json ]] && mv config_bak.json config.json", shell = True) == 0),"> unable to make a copy of configuration file"

