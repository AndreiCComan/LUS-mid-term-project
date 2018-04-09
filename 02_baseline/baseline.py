
# coding: utf-8

# ![DISI](../resources/DISI.jpeg)

# ### Import

# In[1]:


import json
import numpy as np
import operator
import subprocess


# ### Load JSON configuration file

# In[2]:


json_configuration_file = open("config.json", "r", encoding="utf8")
config = json.load(json_configuration_file)


# In[3]:


TRAIN_FILE = "NLSPARQL.train.data"
TEST_FILE = "NLSPARQL.test.data"
BASELINE = config["baseline"]

OUTPUT_DIR = ""


# In[4]:


# make sure the parameter assumes the right values
assert(BASELINE == "random" or       BASELINE == "chance" or       BASELINE == "majority"),"> BASELINE value must be either <random>, <chance>, or <majority>, provided value is <{0}>".format(BASELINE)


# In[5]:


OUTPUT_DIR += "{0}".format(BASELINE)
RESULT_FILE = "{0}/result.txt".format(OUTPUT_DIR)
EVALUATION_FILE = "{0}/evaluation.txt".format(OUTPUT_DIR)


# In[6]:


# remove the output directory if already exists
assert(subprocess.call("rm -rf {0}".format(OUTPUT_DIR), shell = True) == 0),"> unable to remove {0}".format(OUTPUT_DIR)

# make another directory with the specified output directory name
assert(subprocess.call("mkdir -p {0}".format(OUTPUT_DIR), shell = True) == 0),"> unable to mkdir {0}".format(OUTPUT_DIR)


# ### Computing train concepts probabilities

# In[7]:


train_file = open(TRAIN_FILE, "r", encoding="utf8")

total_concepts_count = 0
concept_including_prefix_count_dictionary = {}

for line in train_file:
    line = line.replace("\n", "")
    
    if len(line) == 0: #check if end of sentence
        continue
    
    line_split = line.split("\t")
    assert(len(line_split) == 2)
    
    concept_including_prefix = line_split[1]
    
    if concept_including_prefix not in concept_including_prefix_count_dictionary:
        concept_including_prefix_count_dictionary[concept_including_prefix] = 1
    else:
        concept_including_prefix_count_dictionary[concept_including_prefix] += 1
        
    total_concepts_count += 1
    
train_file.close()    


# In[8]:


concepts_list = []
concepts_probabilities = []

for concept, concept_count in concept_including_prefix_count_dictionary.items():
    concepts_list.append(concept)
    concepts_probabilities.append(concept_count/total_concepts_count)


# ### Generating the result file

# In[9]:


result_file = open(RESULT_FILE, "w", encoding="utf8")
test_file = open(TEST_FILE, "r", encoding="utf8")

for line in test_file:
    line = line.replace("\n", "")
    
    if len(line) == 0: #check if end of sentence
        result_file.write("\n")
        continue
    
    line_split = line.split("\t")
    assert(len(line_split) == 2)
    
    token = line_split[0]
    concept_including_prefix = line_split[1]
    
    predicted_concept = ""
    
    if BASELINE == "random":
        predicted_concept = np.random.choice(concepts_list, 1)[0]
    elif BASELINE == "chance":
        predicted_concept = np.random.choice(concepts_list, 1, p=concepts_probabilities)[0]
    elif BASELINE == "majority":
        predicted_concept = max(concept_including_prefix_count_dictionary.items(), key=operator.itemgetter(1))[0]
    
    assert(predicted_concept != "")
    
    result_file.write("{0} {1} {2}\n".format(token, concept_including_prefix, predicted_concept))
    
result_file.close()
test_file.close()


# ### Generating the evaluation file

# In[10]:


assert(subprocess.call("./conlleval.pl < {0} > {1}".format(RESULT_FILE, EVALUATION_FILE), shell = True) == 0)

