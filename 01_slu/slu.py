
# coding: utf-8

# ![DISI](../resources/DISI.jpeg)

# ### Import

# In[ ]:


# import the json library for reading the configuration file
import json

# import the subprocess library for making command line calls
import subprocess

# import the math library for doing math operations :)
import math

# import the operator which will be used to sort dictionaries
from operator import itemgetter


# ### Load JSON configuration file

# In[ ]:


# set the configuration file
CONFIGURATION_FILE = "config.json"

# open the configuration file
json_configuration_file = open(CONFIGURATION_FILE, "r", encoding="utf8")

# load the configuration file
CONFIGURATION = json.load(json_configuration_file)


# In[ ]:


# read the configuration file parameters
TRAIN_FILE = CONFIGURATION["train_file"]
TRAIN_FEATS_FILE = CONFIGURATION["train_feats_file"]
TEST_FILE = CONFIGURATION["test_file"]
TEST_FEATS_FILE = CONFIGURATION["test_feats_file"]
ADDITIONAL_FEATURE = CONFIGURATION["additional_feature"]
IMPROVEMENT = CONFIGURATION["improvement"]
SMOOTHING = CONFIGURATION["smoothing"]
HANDLE_UNK = CONFIGURATION["handle_unk"]
NGRAM_SIZE = int(CONFIGURATION["ngram_size"])
BACKOFF = CONFIGURATION["backoff"]
BINS = int(CONFIGURATION["bins"])
WITTEN_BELL_K = float(CONFIGURATION["witten_bell_k"])
DISCOUNT_D = float(CONFIGURATION["discount_D"])

# this variable is used to generate the ouput directory name
OUTPUT_DIR = CONFIGURATION["output_dir"]


# ### Check configurations

# In[ ]:


# make sure the parameter assumes the right values
assert(ADDITIONAL_FEATURE == "none" or       ADDITIONAL_FEATURE == "tokenpos" or       ADDITIONAL_FEATURE == "lemmapos" or       ADDITIONAL_FEATURE == "lemma"),"> ADDITIONAL_FEATURE value must be either <none>, <tokenpos>, <lemmapos> or <lemma>, provided value is <{0}>".format(ADDITIONAL_FEATURE)


# In[ ]:


# make sure the parameter assumes the right values
assert(IMPROVEMENT == "none" or       IMPROVEMENT == "wise" or       IMPROVEMENT == "naive"),"> IMPROVEMENTS value must be either <none>, <wise> or <naive>, provided value is <{0}>".format(IMPROVEMENT)


# In[ ]:


# make sure the parameter assumes the right values
assert(SMOOTHING == "witten_bell" or       SMOOTHING == "katz" or       SMOOTHING == "kneser_ney" or       SMOOTHING == "absolute" or       SMOOTHING == "presmoothed" or       SMOOTHING == "unsmoothed"),"> SMOOTHING value must be either <witten_bell>, <katz>, <kneser_ney>, <absolute>, <presmoothed> or <unsmoothed> provided value is <{0}>".format(SMOOTHING)


# In[ ]:


# make sure the parameter assumes the right values
assert(BACKOFF == "true" or       BACKOFF == "false"),"> BACKOFF value must be either <true> or <false>, provided value is <{0}>".format(BACKOFF)


# In[ ]:


# make sure the parameter assumes the right values
assert(NGRAM_SIZE >= 1),"> NGRAM_SIZE must be greater than or equal to 1, provided value is <{0}>".format(NGRAM_SIZE)


# In[ ]:


# make sure the parameter assumes the right values
assert(HANDLE_UNK == "uniform" or       HANDLE_UNK.startswith("cut_off_")),"> HANDLE_UNK value must be either <uniform> or <cut_off_#>, provided value is <{0}>".format(HANDLE_UNK)


# In[ ]:


# generate the output directory name
OUTPUT_DIR += "/AF_{0}_IM_{1}_SM_{2}_NS_{3}_HU_{4}"              .format(ADDITIONAL_FEATURE, IMPROVEMENT, SMOOTHING, NGRAM_SIZE, HANDLE_UNK)


# ### Helper functions

# In[ ]:


# this function takes as input:
#     * input file from which we take the concept
#     * features file from which we take the POS-tag or the lemma
#     * parameter that decides which feature (POS-tag or lemma) to take into account and the way it will be used
# it returns the path of the generated file
def apply_additional_feature(input_file_path, features_file_path, additional_feature):
    
    print("> applying {0} additional feature ...".format(additional_feature))
    
    # define the path for the output file
    output_file_path = "{0}_{1}.txt".format(input_file_path.split(".")[0], additional_feature)
    
    # open the input file (read mode)
    input_file = open(input_file_path, "r", encoding="utf8")
    
    # open the features file (read mode)
    features_file = open(features_file_path, "r", encoding="utf8")
    
    # open the output file (write mode)
    output_file = open(output_file_path, "w", encoding="utf8")
    
    # list for the input file lines
    input_file_lines = []
    
    # list for the features file lines
    features_file_lines = []
    
    # iterate over each line within the input file
    for line in input_file:
        
        # get rid of the newline character
        line = line.replace("\n", "")
        
        # append the line in the input file lines list
        input_file_lines.append(line)
    
    # iterate over each line within the features file
    for line in features_file:
        
        # get rid of the newline character
        line = line.replace("\n", "")
        
        # append the line in the features file lines list
        features_file_lines.append(line)
    
    # make sure the input file and features file number of lines match
    assert(len(input_file_lines) == len(features_file_lines)),    "> length of the two files should match"
    
    # for each line in the output file
    for line_index in range(len(input_file_lines)):
        
        # take the line from the input file
        input_file_line = input_file_lines[line_index]
        
        # take the line from the features file
        features_file_line = features_file_lines[line_index]
        
        # if end of sentence in the input file (blank line)
        if len(input_file_line) == 0:
            
            # makes sure is the same for the features file
            assert(len(features_file_line) == 0),            "> line should be empty"
            
            # then write the blank line to the output file
            output_file.write("\n")
            
            # go to the next line
            continue
        
        # if not end of sentences, split the input file line
        input_file_line_split = input_file_line.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(input_file_line_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(input_file_line_split))
        
        # split the features file line
        features_file_line_split = features_file_line.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(features_file_line_split) == 3),        "> expected 3 tab separated values, found {0} instead".format(len(features_file_line_split))
        
        # take the token from the input file line
        input_file_token = input_file_line_split[0]
        
        # take the concept from the input file line
        input_file_concept = input_file_line_split[1]
        
        # take the token from the features file line
        features_file_token = features_file_line_split[0]
        
        # take the POS-tag from the features file line
        features_file_pos = features_file_line_split[1]
        
        # take the lemma from the features file line
        features_file_lemma = features_file_line_split[2]
        
        # make sure the input token and features token match
        assert(input_file_token == features_file_token),        "> token should match between the two file"
        
        # check which feature should be used and generate the token for the output file
        if additional_feature == "tokenpos":
            
            # create the tokenpos token
            tokenpos = "{0}_{1}".format(input_file_token, features_file_pos)
            
            # write the tokenpos token and the input file concept to the output file
            output_file.write("{0}\t{1}\n".format(tokenpos, input_file_concept))
            
        elif additional_feature == "lemmapos":
            
            # create the lemmapos token
            lemmapos = "{0}_{1}".format(features_file_lemma, features_file_pos)
            
            # write the lemmapos token and the input file concept to the output file
            output_file.write("{0}\t{1}\n".format(lemmapos, input_file_concept))
            
        else: # additional_feature == "lemma":
            
            # write the lemma token and the input file concept to the output file
            output_file.write("{0}\t{1}\n".format(features_file_lemma, input_file_concept))
    
    # close the input file
    input_file.close()
    
    # close the features file
    features_file.close()
    
    # close the output file
    output_file.close()
    
    # return the output file path
    return output_file_path


# In[ ]:


# this function takes as input a file and generates a modified version of it by following this example:
#
# the following sentence
#
#       show	O
#       credits	O
#       for	O
#       the	B-movie.name
#       godfather	I-movie.name
#
# becomes
#
#       show	O
#       credits	O
#       for	_for
#       the	B-movie.name
#       godfather	I-movie.name
#
# it returns the path of the generated file
def apply_wise_improvement(input_file_path):
    
    print("> applying wise improvement ...")
    
    # define the path for the output file
    output_file_path = "{0}_wise.txt".format(input_file_path.split(".")[0])
    
    # open the output file (write mode)
    output_file = open(output_file_path, "w", encoding="utf8")
    
    # open the input file (read mode)
    input_file = open(input_file_path, "r", encoding="utf8")
    
    # list for the input file lines
    input_file_lines = []
    
    # iterate over each line within the input file
    for line in input_file:
        
        # get rid of the newline character
        line = line.replace("\n", "")
        
        # append the line in the input file lines list
        input_file_lines.append(line)
    
    # iterate over each line in the input file lines list
    for current_line_index in range(len(input_file_lines)):
        
        # get the next line index
        next_line_index = current_line_index + 1
        
        # retrieve the next line
        current_line = input_file_lines[current_line_index]
        
        # check for lower boundary
        # check for upper boundary
        # check for blank line (end of sentence)
        if (current_line_index == 0) or (next_line_index > len(input_file_lines)) or (len(current_line) == 0):
                
                # write blank line to the output file
                output_file.write("{0}\n".format(current_line))
                
                # go to the next line
                continue
        
        # if not end of sentences, split the current line
        current_line_split = current_line.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(current_line_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(current_line_split))
        
        # take the token from the current line
        current_line_token = current_line_split[0]
        
        # take the concept from the current line
        current_line_concept= current_line_split[1]
        
        # if the current line concept is an O
        if current_line_concept == "O":
            
            # retrieve the next line
            next_line = input_file_lines[next_line_index]
            
            # if next line is a blank line
            if (len(next_line) == 0):
                
                # write the current line to the output file
                output_file.write("{0}\n".format(current_line))
                
            else: # if not a blank line
                
                # split the next line
                next_line_split = next_line.split("\t")
                
                # make sure the split produces the expected number of arguments
                assert(len(next_line_split) == 2),                "> expected 2 tab separated values, found {0} instead".format(len(next_line_split))
                
                # take the concept from the next line
                next_line_concept = next_line_split[1]
                
                # check if next line concept is an O
                if (next_line_concept == "O"):
                    
                    # if it is then write the current line to the output file
                    output_file.write("{0}\n".format(current_line))
                    
                else: # if next line concept not an O
                    
                    # write the current line token and as token, and the current line token preceded by an underscore as concept
                    output_file.write("{0}\t_{0}\n".format(current_line_token))
                    
        else: # if current line concept is not an O
            
            # write the current line to the output file
            output_file.write("{0}\n".format(current_line))
    
    # close the input file
    input_file.close()
    
    # close the output file
    output_file.close()
    
    # return the output file path
    return output_file_path


# In[ ]:


# this function takes as input a file and generates a modified version of it by following this example:
#
# the following sentence
#
#       show	O
#       credits	O
#       for	O
#       the	B-movie.name
#       godfather	I-movie.name
#
# becomes
#
#       show	_show
#       credits	_credits
#       for	_for
#       the	B-movie.name
#       godfather	I-movie.name
#
# it returns the path of the generated file
def apply_naive_improvement(input_file_path):
    
    # define the path for the output file
    output_file_path = "{0}_naive.txt".format(input_file_path.split(".")[0])
    
    # open the output file (write mode)
    output_file = open(output_file_path, "w", encoding="utf8")
    
    # open the input file (read mode)
    input_file = open(input_file_path, "r", encoding="utf8")
    
    # iterate over each line within the input file
    for line in input_file:
        
        # get rid of the newline character
        line = line.replace("\n", "")
        
        # if end of sentence (blank line)
        if len(line) == 0:
            
            # write black line to the output file
            output_file.write("\n")
            
            # go to the next line
            continue
        
        # if not blank line, split it
        line_split = line.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(line_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(line_split))
        
        # take the token from the line
        token = line_split[0]
        
        # take the concept from the line
        concept = line_split[1]
        
        # check if concept is an O
        if concept == "O":
            
            # write token as token, and the token preceded by an underscore as concept to the output file
            output_file.write("{0}\t_{0}\n".format(token))
        
        else:
            
            # write the line to the output file
            output_file.write("{0}\t{1}\n".format(token, concept))
    
    # close input file
    input_file.close()
    
    # close output file
    output_file.close()
    
    # return the output file path
    return output_file_path


# In[ ]:


# this function takes an input a file and generates different files, such as:
#     * token counts file
#     * concept counts file
#     * token-concept counts file
#     * token-concept probabilities file
# it returns some useful dictionaries
def generate_useful_files(train_file_path, output_dir):
    
    print("> generating useful files ...")
    
    # define the path for the token counts file
    token_counts_file_path = "{0}/token.counts".format(output_dir)
    
    # define the path for the concept counts file
    concept_counts_file_path = "{0}/concept.counts".format(output_dir)
    
    # define the path for the token-concept counts file
    token_concept_counts_file_path = "{0}/token_concept.counts".format(output_dir)
    
    # define the path for the token-concept probabilities file
    token_concept_probs_file_path = "{0}/token_concept.probs".format(output_dir)
    
    # open the train file (read mode)
    train_file = open(train_file_path, "r", encoding="utf8")
    
    # open the token counts file (write mode)
    token_counts_file = open(token_counts_file_path, "w", encoding="utf8")
    
    # open the concept counts file (write mode)
    concept_counts_file = open(concept_counts_file_path, "w", encoding="utf8")
    
    # open the token-cocept counts file (write mode)
    token_concept_counts_file = open(token_concept_counts_file_path, "w", encoding="utf8")
    
    # open the token-concept probabilities file (write mode)
    token_concept_probs_file = open(token_concept_probs_file_path, "w", encoding="utf8")
    
    # declare the token counts dictionary
    token_counts_dictionary = {}
    
    # declare the concept counts dictionary
    concept_counts_dictionary = {}
    
    # declare the token-concept counts dictionary
    token_concept_counts_dictionary = {}
    
    # declare the token-concept probabilities dictionary
    token_concept_probs_dictionary = {}
    
    # iterate over each line in the train file
    for line in train_file:
        
        # get rid of the newline character
        line = line.replace("\n", "")
        
        # if end of sentence (blank line)
        if len(line) == 0:
            
            # go to the next line
            continue
        
        # if not end of sentence, split the line
        line_split = line.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(line_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(line_split))
        
        # take the token from the line
        token = line_split[0]
        
        # take the concept from the line
        concept = line_split[1]
        
        # create the token-concept pair
        token_concept = "{0}\t{1}".format(token, concept)
        
        # check if token was already inserted within the token counts dictionary
        if token not in token_counts_dictionary:
        
            # if not then add it to the dictionary
            token_counts_dictionary[token] = 1
        
        else: # if it was already inserted
            
            # increment the count by 1
            token_counts_dictionary[token] += 1
        
        # check if concept was already inserted within the concept counts dictionary
        if concept not in concept_counts_dictionary:
            
            # if not then add it to the dictionary
            concept_counts_dictionary[concept] = 1
        
        else: # if it was already inserted
            
            # increment the count by 1
            concept_counts_dictionary[concept] += 1
        
        # check if token-concept was already inserted within the token-concept counts dictionary
        if token_concept not in token_concept_counts_dictionary:
            
            # if not then add it to the dictionary
            token_concept_counts_dictionary[token_concept] = 1
        
        else: # if it was already inserted
            
            # increment the count by 1
            token_concept_counts_dictionary[token_concept] += 1
    
    # compute token-concept probabilities and populate the token-concept proabilities dictionary
    for token_concept in token_concept_counts_dictionary:
        
        # split the token concept pair
        token_concept_split = token_concept.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(token_concept_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(token_concept_split))
        
        # take the concept from the token-concept pair
        concept = token_concept_split[1]
        
        # compute the token-concept probability
        token_concept_prob = -math.log(float(token_concept_counts_dictionary[token_concept]) / float(concept_counts_dictionary[concept]))
        
        # add the token-concept probability to the dictionary
        token_concept_probs_dictionary[token_concept] = token_concept_prob
    
    # sort (by value) the token counts dictionary
    sorted_token_counts_list = sorted(token_counts_dictionary.items(), key = itemgetter(1), reverse = True)
    
    # sort (by value) the concept counts dictionary
    sorted_concept_counts_list = sorted(concept_counts_dictionary.items(), key = itemgetter(1), reverse = True)
    
    # sort (by value) the token-concept counts dictionary
    sorted_token_concept_counts_list = sorted(token_concept_counts_dictionary.items(), key = itemgetter(1), reverse = True)
    
    # sort (by value) the token-concept probabilities dictionary
    sorted_token_concept_probs_list = sorted(token_concept_probs_dictionary.items(), key = itemgetter(1), reverse = True)
    
    # write the sorted token counts dictionary to file
    for token, token_count in sorted_token_counts_list:
        token_counts_file.write("{0}\t{1}\n".format(token, token_count))
    
    # write the sorted concept counts dictionary to file
    for concept, concept_count in sorted_concept_counts_list:
        concept_counts_file.write("{0}\t{1}\n".format(concept, concept_count))
        
    # write the sorted token-concept counts dictionary to file
    for token_concept, token_concept_count in sorted_token_concept_counts_list:
        token_concept_counts_file.write("{0}\t{1}\n".format(token_concept, token_concept_count))

    # write the sorted token-concept probabilities dictionary to file
    for token_concept, token_concept_prob in sorted_token_concept_probs_list:
        token_concept_probs_file.write("{0}\t{1}\n".format(token_concept, token_concept_prob))
    
    # close train file
    train_file.close()
    
    # close the token counts file
    token_counts_file.close()
    
    # close the concept counts file
    concept_counts_file.close()
    
    # close the token-concept counts file
    token_concept_counts_file.close()
    
    # close the token-concept probabilities file
    token_concept_probs_file.close()    
    
    # return dictionaries
    return (token_counts_dictionary,            concept_counts_dictionary,            token_concept_counts_dictionary,            token_concept_probs_dictionary)


# In[ ]:


# this function takes as input a file and generates a lexicon and transducer in the passed output directory path
# handle unknown-concept with a uniform probability approach
def generate_lexicon_and_transducer_uniform(train_file_path, output_dir):
    
    print("> generating lexicon and transducer uniform ...")
    
    # define the path for the lexicon file
    lexicon_file_path = "{0}/lexicon.lex".format(output_dir)
    
    # define the path for the transducer file
    transducer_file_path = "{0}/transducer_uniform.txt".format(output_dir)
    
    # open the lexicon file (write mode)
    lexicon_file = open(lexicon_file_path, "w", encoding="utf8")
    
    # open the transducer file (write mode)
    transducer_file = open(transducer_file_path, "w", encoding="utf8")
    
    # get dictionaries
    token_counts_dictionary,    concept_counts_dictionary,    token_concept_counts_dictionary,    token_concept_probs_dictionary = generate_useful_files(train_file_path, output_dir)
    
    # GENERATE LEXICON
    
    # lexeme list
    lexeme_list = []
    
    # lexeme index
    lexeme_index = 0
    
    # add to the lexeme list the first lexeme
    lexeme_list.append("<epsilon>")
    
    # write the first lexeme to the lexicon file
    lexicon_file.write("<epsilon>\t{0}\n".format(lexeme_index))
    
    # increment the lexeme index
    lexeme_index += 1
    
    # iterate over each token
    for token, token_count in token_counts_dictionary.items():
        
        # check if token was already inserted in the lexeme list
        if token not in lexeme_list:
            
            # if not then add it to the lexeme list
            lexeme_list.append(token)
            
            # write it to the lexicon file
            lexicon_file.write("{0}\t{1}\n".format(token, lexeme_index))
            
            # increment the lexeme index
            lexeme_index +=1
    
    # iterate over each concept
    for concept, concept_count in concept_counts_dictionary.items():
        
        # check if concept was alreary inserted in the lexeme list
        if concept not in lexeme_list:
            
            # if not then add it ot the lexeme list
            lexeme_list.append(concept)
            
            # write it to the lexicon file
            lexicon_file.write("{0}\t{1}\n".format(concept, lexeme_index))
            
            # increment the lexeme index
            lexeme_index +=1
    
    # add to the lexeme list the last lexeme
    lexeme_list.append("<unk>")

    # write the last lexeme to the lexicon file
    lexicon_file.write("<unk>\t{0}\n".format(lexeme_index))
    
    # increment the lexeme index
    lexeme_index += 1
           
    # GENERATE TRANSDUCER
    
    # devine the uniform probability of the unkown-concept pair
    unk_concept_prob = -math.log(1/len(concept_counts_dictionary))
    
    # iterate over each token-concept probability pair
    for token_concept, token_concept_prob in token_concept_probs_dictionary.items():
        
        # split the token-concept pair
        token_concept_split = token_concept.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(token_concept_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(token_concept_split))
        
        # take the token from the token-concept pair
        token = token_concept_split[0]
        
        # take the concept from the token-concept pair
        concept = token_concept_split[1]
        
        # write to the transducer file the token concept transition with the corresponding probability
        transducer_file.write("0\t0\t{0}\t{1}\t{2}\n".format(token, concept, token_concept_prob))
     
    # iterate over each concept
    for concept, concept_count in concept_counts_dictionary.items():
        
        # write to the transducer file the unkown-concept transition with the corresponding probability
        transducer_file.write("0\t0\t{0}\t{1}\t{2}\n".format("<unk>", concept, unk_concept_prob))
    
    # write the final state to the transducer file
    transducer_file.write("0")
    
    # close the lexicon file
    lexicon_file.close()
    
    # close the transducer file
    transducer_file.close()
    
    # return the lexicon file path and transducer file path
    return (lexicon_file_path, transducer_file_path) 


# In[ ]:


# this function takes as input a file and generates a lexicon and transducer in the passed output directory path
# handle unkown-concept with a cut-off approach
#     1) take the token-concept counts < threshold
#     2) group by concept
#     3) accumulate the counts
#     4) divive each concept accumulated counts by the total token-concept counts
#     5) assign this value as unknown-concept probability
def generate_lexicon_and_transducer_cut_off(train_file_path, cut_off_value, output_dir):
    
    print("> generating lexicon and transducer cut-off-{0} ...".format(cut_off_value))
    
    # define the path for the lexicon file
    lexicon_file_path = "{0}/lexicon.lex".format(output_dir)
    
    # define the path for the transducer file
    transducer_file_path = "{0}/transducer_cut_off_{1}.txt".format(output_dir, cut_off_value)
    
    # open the lexicon file (write mode)
    lexicon_file = open(lexicon_file_path, "w", encoding="utf8")
    
    # open the transducer file (write mode)
    transducer_file = open(transducer_file_path, "w", encoding="utf8")

    # get dictionaries
    token_counts_dictionary,    concept_counts_dictionary,    token_concept_counts_dictionary,    token_concept_probs_dictionary = generate_useful_files(train_file_path, output_dir)
    
    # GENERATE TRANSDUCER
    
    # token-concept counts over threshold dictionary
    extracted_token_concept_counts_dictionary = {}
    
    # concept accumulated counts dictionary
    concepts_under_threshold_counts_dictionary = {}
    
    # total token-concept counts
    total_token_concept_counts = 0
    
    # iterate over each token-concept count
    for token_concept, token_concept_count in token_concept_counts_dictionary.items():
        
        # split the token-concept pair
        token_concept_split = token_concept.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(token_concept_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(token_concept_split))
        
        # take the token from the token-concept pair
        token = token_concept_split[0]
        
        # take the concept from the token-concept pair
        concept = token_concept_split[1]
        
        # add the token-concept count to the total token-concept counts
        total_token_concept_counts += token_concept_count
        
        # check if the token-concept count is under the threshold
        if token_concept_count <= cut_off_value:
                
                # if it is then check if the concept was already inserted in the concepts under threshold dictionary
                if concept not in concepts_under_threshold_counts_dictionary:
                    
                    # if not then add it with the corresponding token-concept count
                    concepts_under_threshold_counts_dictionary[concept] = token_concept_count
                
                else: # if it was already inserted
                    
                    # accumulate the token-concept count for this concept
                    concepts_under_threshold_counts_dictionary[concept] += token_concept_count
                    
        else: # if token-concept count is over the threshold
            
            # add the token-concept pair with the corresponding count to the token-concept over the threshold dictionary
            extracted_token_concept_counts_dictionary[token_concept] = token_concept_count
    
    # iterate over each token-concept pair within the token-concept over the threshold dictionary
    for token_concept, token_concept_count in extracted_token_concept_counts_dictionary.items():
        
        # split the token-concept pair
        token_concept_split = token_concept.split("\t")

        # make sure the split produces the expected number of arguments
        assert(len(token_concept_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(token_concept_split))

        # take the token from the token-concept pair
        token = token_concept_split[0]
        
        # take the concept from the token-concept pair
        concept = token_concept_split[1]
        
        # compute the probability for the token-concept pair
        token_concept_prob = -math.log(float(extracted_token_concept_counts_dictionary[token_concept]) / float(concept_counts_dictionary[concept]))
        
        # write to the transducer file the token concept transition with the corresponding probability
        transducer_file.write("0\t0\t{0}\t{1}\t{2}\n".format(token, concept, token_concept_prob))
    
    # iterate over each concept under threshold
    for concept_under_threshold, concept_under_threshold_count in concepts_under_threshold_counts_dictionary.items():
        
        # define the unkown word
        unk = "<unk>"
        
        # take the concept under threshold
        concept = concept_under_threshold
        
        # compute the unkown-concept probability
        unk_concept_prob = -math.log(concept_under_threshold_count / total_token_concept_counts)
        
        # write to the transducer file the unkown-concept transition with the corresponding probability
        transducer_file.write("0\t0\t{0}\t{1}\t{2}\n".format(unk, concept, unk_concept_prob))
    
    # write the final state to the transducer file
    transducer_file.write("0")
    
    # GENERATE LEXICON
    
    # lexeme list
    lexeme_list = []
    
    # lexeme index
    lexeme_index = 0
    
    # add to the lexeme list the first lexeme
    lexeme_list.append("<epsilon>")
    
    # write the first lexeme to the lexicon file
    lexicon_file.write("<epsilon>\t{0}\n".format(lexeme_index))
    
    # increment the lexeme index
    lexeme_index += 1
    
    # iterate over each token-concept pair
    for token_concept, token_concept_count in extracted_token_concept_counts_dictionary.items():
        
        # split the token-concept pair
        token_concept_split = token_concept.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(token_concept_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(token_concept_split))
        
        # take the token from the token-concept pair
        token = token_concept_split[0]
        
        # take the concept from the token-concept pair
        concept = token_concept_split[1]
        
        # check if token was already inserted in the lexeme list
        if token not in lexeme_list:
            
            # if not then add it
            lexeme_list.append(token)
            
            # write it to the lexicon file
            lexicon_file.write("{0}\t{1}\n".format(token, lexeme_index))
            
            # increment the lexeme index
            lexeme_index += 1
            
        # check if concept was already inserted in the lexeme list    
        if concept not in lexeme_list:
            
            # if not then add it
            lexeme_list.append(concept)
            
            # write it to the lexicon file
            lexicon_file.write("{0}\t{1}\n".format(concept, lexeme_index))
            
            # increment the lexeme index
            lexeme_index += 1
    
    # iterate over each concept under threshold
    for concept_under_threshold, concept_under_threshold_count in concepts_under_threshold_counts_dictionary.items():
        
        # check if concept under threshold was already inserted in the lexeme list
        if concept_under_threshold not in lexeme_list:
            
            # if not then add it
            lexeme_list.append(concept_under_threshold)
            
            # write it to the lexicon file
            lexicon_file.write("{0}\t{1}\n".format(concept_under_threshold, lexeme_index))
            
            # increment the lexeme index
            lexeme_index += 1
    
    # add to the lexeme list the last lexeme
    lexeme_list.append("<unk>")
    
    # write the last lexeme to the lexicon file
    lexicon_file.write("<unk>\t{0}\n".format(lexeme_index))
    
    # increment the lexeme index
    lexeme_index += 1
    
    # close the lexicon file
    lexicon_file.close()
    
    # close the transducer file
    transducer_file.close()
    
    # return the lexicon file path and transducer file path
    return (lexicon_file_path, transducer_file_path) 


# In[ ]:


# this is a helper function that decides which specific function should be called
def generate_lexicon_and_transducer(train_file_path, handle_unk, output_dir):
    
    # handle unknown-concept with a uniform probability approach
    if handle_unk == "uniform":
        
        # call the specific function
        return generate_lexicon_and_transducer_uniform(train_file_path, output_dir)
    
    else: # handle unknown-concept with a cut-off approach
        
        # retrieve the cut-off threshold
        cut_off_value = int(handle_unk.replace("cut_off_", ""))
        
        # call the specific function
        return generate_lexicon_and_transducer_cut_off(train_file_path, cut_off_value, output_dir)


# In[ ]:


# this function compiles a transducer in the passed output directory,
#      by taking a lexicon and transducer (text format) as input
def compile_transducer(lexicon_file_path, transducer_file_path, output_dir):
    
    print("> compiling transducer ...")
    
    # define the trasnducer path
    transducer_fst_file_path = "{0}/transducer.fst".format(output_dir)
    
    # call the command line
    assert(subprocess.call("fstcompile --isymbols={0} --osymbols={0} {1} | fstarcsort > {2}"
                           .format(lexicon_file_path,\
                                   transducer_file_path,\
                                   transducer_fst_file_path), shell = True) == 0)
    
    # return the compiled transducer file path
    return transducer_fst_file_path


# In[ ]:


# this function extracts the concepts from sentences and outputs them 
#      in a file specified in the passed output directory
def create_concept_sentences(train_file_path, output_dir):
    
    print("> creating concept sentences ...")
    
    # define the concept sentences file path
    concept_sentences_file_path = "{0}/concept_sentences.txt".format(output_dir)
    
    # open the train file (read mode)
    train_file = open(train_file_path, "r", encoding="utf8")
    
    # open the concept sentences file (write mode)
    concept_sentences_file = open(concept_sentences_file_path, "w", encoding="utf8")
    
    # list of concepts for a single sentence
    sentence_concept_list = []
    
    # iterate over each line in the train file
    for line in train_file:
        
        # get rid of the newline character
        line = line.replace("\n", "")
        
        # check for end of sentence (blank line)
        if len(line) == 0:
            
            # ceate the sentence from the list of concepts
            sentence = " ".join(sentence_concept_list)
            
            # write the sentence to the concept sentences file
            concept_sentences_file.write("{0}\n".format(sentence))
            
            # empty the list of concepts for a single sentence
            sentence_concept_list = []
            
            # go to the next line
            continue
        
        # if not end of sentence, split the line
        line_split = line.split("\t")
        
        # make sure the split produces the expected number of arguments
        assert(len(line_split) == 2),        "> expected 2 tab separated values, found {0} instead".format(len(line_split))
        
        # get the concept from the line
        concept = line_split[1]
        
        # insert the concept in the list of concepts for a single sentence
        sentence_concept_list.append(concept)
        
    # close the train file
    train_file.close()
    
    # close the concept sentences file
    concept_sentences_file.close()
    
    # return the concept sentences file path
    return concept_sentences_file_path


# In[ ]:


# this function creates the language model
def create_language_model(lexicon_file_path, concept_sentences_file_path, ngram_size, smoothing, backoff, bins, witten_bell_k, discount_D, output_dir):
    
    print("> creating language model ...")
    
    # define the language model far format file path
    language_model_far_file_path = "{0}/language_model.far".format(output_dir)
    
    # define the language model cnt format file path
    language_model_cnt_file_path = "{0}/language_model.cnt".format(output_dir)
    
    # define the language model lm format file path
    language_model_lm_file_path = "{0}/language_model.lm".format(output_dir)
    
    # generate the language model far format file
    assert(subprocess.call("farcompilestrings --symbols={0} --unknown_symbol='<unk>' {1} > {2}"
           .format(lexicon_file_path, concept_sentences_file_path, language_model_far_file_path), shell = True) == 0)
    
    # generate the language model cnt format file
    assert(subprocess.call("ngramcount --order={0} --require_symbols=false {1} > {2}"
           .format(ngram_size, language_model_far_file_path, language_model_cnt_file_path), shell = True) == 0)
    
    # generate the language model lm format file
    assert(subprocess.call("ngrammake --method={0} --backoff={1} --bins={2} --witten_bell_k={3} --discount_D={4} {5} > {6}"
           .format(smoothing, backoff, bins, witten_bell_k, discount_D, language_model_cnt_file_path, language_model_lm_file_path), shell = True) == 0)
    
    # return the language model lm format file path
    return language_model_lm_file_path    


# In[ ]:


# this function compiles (creates a trasnducer of) a string given as input
def compile_string(sentence, lexicon_file_path, output_dir):
        
    # define the compiled string file path
    result_compiled_string_file_path = "{0}/compiled_string.fst".format(output_dir)
    
    # compile the string
    assert(subprocess.call("echo \"{0}\" | farcompilestrings --symbols={1} --unknown_symbol='<unk>' --generate_keys=1 --keep_symbols | farextract --filename_suffix='.fst'"           .format(sentence, lexicon_file_path), shell = True) == 0),    "> unable to compile string"
    
    # move the compiled string to the corresponding directory
    assert(subprocess.call("mv 1.fst {0}".format(result_compiled_string_file_path), shell = True) == 0),    "> unable to move file"
    
    # return the compiled string file path
    return result_compiled_string_file_path


# In[ ]:


# this function does the composition between:
#     * compiled_string
#     * transducer
#     * language model
def execute_composition(compiled_string_file_path, transducer_fst_file_path, language_model_lm_file_path, lexicon_file_path, output_dir):
        
    # define the composition file path
    result_composition_file_path = "{0}/composition_result.txt".format(output_dir)
    
    # call the composition command
    assert(subprocess.call("fstcompose {0} {1} | fstcompose - {2} | fstrmepsilon | fstshortestpath | fsttopsort | fstprint --isymbols={3} --osymbols={3} > {4}"                           .format(compiled_string_file_path,                                   transducer_fst_file_path,                                   language_model_lm_file_path,                                   lexicon_file_path,                                   result_composition_file_path), shell = True) == 0),    "> unable to compose"
    
    # return the composition file path
    return result_composition_file_path


# In[ ]:


# this function tags a file
def tag_file(test_file_path, lexicon_file_path, transducer_fst_file_path, language_model_lm_file_path, output_dir):
    
    print("> tagging file ...")
    
    # define the tagged file path
    result_tagged_file_path = "{0}/result_tagged_file.txt".format(output_dir)
    
    # open the tagged file (write mode)
    result_tagged_file = open(result_tagged_file_path, "w", encoding="utf8")
    
    # open the test file
    test_file = open(test_file_path, "r", encoding="utf8")
    
    # number of tagged senteces count
    sentences_tagged_count = 0
    
    # list of tokens per sentence in the test file
    list_of_test_tokens = []
    
    # list of concepts per sentence in the test file
    list_of_test_concepts = []
    
    # iterate over each line in the test file
    for line in test_file:
        
        # get rid of the newline character
        line = line.replace("\n", "")
        
        # check if end of sentece (blank line)
        if len(line) == 0:
            
            # make sure the lenght of test tokens and concepts match
            assert(len(list_of_test_tokens) == len(list_of_test_concepts)),            "> tokens list and concepts list length should match"
            
            # create the sentence from tokens
            sentence = " ".join(list_of_test_tokens)
            
            # compile the sentences
            compiled_string_file_path = compile_string(sentence, lexicon_file_path, output_dir)
            
            # compose the sentence with the transducer and the language model
            composition_result_file_path = execute_composition(compiled_string_file_path,                                                               transducer_fst_file_path,                                                               language_model_lm_file_path,                                                               lexicon_file_path,                                                               output_dir)
            
            # open the composition file
            composition_result_file = open(composition_result_file_path, "r", encoding="utf8")
            
            # list of tokens per sentence in the composition file
            list_of_result_tokens = []
            
            # list of concepts per sentence in the composition file
            list_of_result_concepts = []
            
            # iterate over each line in the composition file
            for line in composition_result_file:
                
                # get rid of the newline character
                line = line.replace("\n", "")
                
                # split the line
                line_split = line.split("\t")
                
                # make sure the split produces the expected number of arguments
                assert((len(line_split) == 5 or len(line_split) == 2)),                "> expected 5 OR 2 tab separated values, found {0} instead".format(len(line_split))
                
                # check if the read line is not a final state
                if (len(line_split) == 5): # not final state
                    
                    # take the token from the line
                    result_token = line_split[2]
                    
                    # take the concept from the line
                    result_concept = line_split[3]
                    
                    # insert the token in the list of tokens per sentence in the composition file
                    list_of_result_tokens.append(result_token)
                    
                    # insert the concept in the list of concepts per sentence in the composition file
                    list_of_result_concepts.append(result_concept)
            
            # close the composition file
            composition_result_file.close()
            
            # make sure the number of test and composition tokens match
            assert(len(list_of_test_tokens) == len(list_of_result_tokens)),            "> length of test tokens does not match with the length of result tokens"
            
            # make sure the number of test and composition concepts match
            assert(len(list_of_test_concepts) == len(list_of_result_concepts)),            "> length of test concepts does not match with the length of result concepts"
            
            # write to the tagged file
            for index in range(len(list_of_test_tokens)):
                
                result_tagged_file.write("{0} {1} {2}\n".format(list_of_test_tokens[index],                                                                list_of_test_concepts[index],                                                                list_of_result_concepts[index]))
            # add end of sentence to the tagged file
            result_tagged_file.write("\n")
            
            # empty the list of tokens in the test file
            list_of_test_tokens = []
            
            # empty the list of concepts in the test file
            list_of_test_concepts = []
            
            # increment the number of tagged sentences
            sentences_tagged_count += 1
            
            # print the current number of analyzed sentences
            print("# of analyzed sentences: {0}".format(sentences_tagged_count), end="\r")
            
        else: # if not end of sentence
            
            # split the line
            line_split = line.split("\t")
            
            # make sure the split produces the expected number of arguments
            assert(len(line_split) == 2),            "> expected 2 tab separated values, found {0} instead".format(len(line_split))
            
            # take the token from the line
            test_token = line_split[0]
            
            # take the concept from the line
            test_concept = line_split[1]
            
            # insert the token in the list of tokens per sentence in the test file 
            list_of_test_tokens.append(test_token)
            
            # insert the concept in the list of concepts per sentence in the test file 
            list_of_test_concepts.append(test_concept)
    
    # close the tagged file
    result_tagged_file.close()
    
    # close the test file
    test_file.close()
    
    # return the tagged file path
    return result_tagged_file_path


# In[ ]:


# this function evaluates the result of the tagging process
def evaluate_file(result_tagged_file_path, output_dir):
    
    print("> evaluating file ...")
    
    # define the evaluation file path
    result_evaluation_file_path = "{0}/evaluation.txt".format(output_dir)
    
    # call the evaluation script
    assert(subprocess.call("./conlleval.pl < {0} > {1}"                           .format(result_tagged_file_path, result_evaluation_file_path), shell = True) == 0)
    
    # return the evaluation file path
    return result_evaluation_file_path


# ### Main

# In[ ]:


# remove the output directory if already exists
assert(subprocess.call("rm -rf {0}".format(OUTPUT_DIR), shell = True) == 0),"> unable to remove {0}".format(OUTPUT_DIR)

# make another directory with the specified output directory name
assert(subprocess.call("mkdir -p {0}".format(OUTPUT_DIR), shell = True) == 0),"> unable to mkdir {0}".format(OUTPUT_DIR)

# copy the config file to the output directoryt
assert(subprocess.call("cp {0} {1}".format(CONFIGURATION_FILE, OUTPUT_DIR), shell = True) == 0),"unable to copy {0} to {1}".format(CONFIGURATION_FILE, OUTPUT_DIR)

# define the train temporary file path
train_temp_file_path = "{0}/{1}".format(OUTPUT_DIR, "train_temp.txt")

# define the test temporary file path
test_temp_file_path = "{0}/{1}".format(OUTPUT_DIR, "test_temp.txt")

# copy the train file to the output directory and rename it
assert(subprocess.call("cp {0} {1}".format(TRAIN_FILE, train_temp_file_path), shell = True) == 0),"unable to copy {0} to {1}".format(TRAIN_FILE, train_temp_file_path)

# copy the test file to the output directory and rename it
assert(subprocess.call("cp {0} {1}".format(TEST_FILE, test_temp_file_path), shell = True) == 0),"unable to copy {0} to {1}".format(TEST_FILE, test_temp_file_path)

# check if additional feature is requested
if ADDITIONAL_FEATURE != "none":
    
    # apply additional feature to the train file
    train_additional_feature_file_path = apply_additional_feature(train_temp_file_path, TRAIN_FEATS_FILE, ADDITIONAL_FEATURE)
    
    # apply additional feature to the test file
    test_additional_feature_file_path = apply_additional_feature(test_temp_file_path, TEST_FEATS_FILE, ADDITIONAL_FEATURE)
    
    # check if improvement is requested
    if IMPROVEMENT == "wise":
        
        # apply improvement to the train file
        train_additional_feature_wise_file_path = apply_wise_improvement(train_additional_feature_file_path)
        
        # apply improvement to the test file
        test_additional_feature_wise_file_path = apply_wise_improvement(test_additional_feature_file_path)
        
        # generate lexicon and transducer
        lexicon_file_path, transducer_file_path = generate_lexicon_and_transducer(train_additional_feature_wise_file_path, HANDLE_UNK, OUTPUT_DIR)
        
        # compile transducer
        transducer_fst_file_path = compile_transducer(lexicon_file_path, transducer_file_path, OUTPUT_DIR)
        
        # generate concept sentences file
        concept_sentences_file_path = create_concept_sentences(train_additional_feature_wise_file_path, OUTPUT_DIR)
        
        # generate language model
        language_model_lm_file_path = create_language_model(lexicon_file_path, concept_sentences_file_path, NGRAM_SIZE, SMOOTHING, BACKOFF, BINS, WITTEN_BELL_K, DISCOUNT_D, OUTPUT_DIR)
        
        # tag the test file
        result_tagged_file_path = tag_file(test_additional_feature_wise_file_path, lexicon_file_path, transducer_fst_file_path, language_model_lm_file_path, OUTPUT_DIR)       
        
        # evaluate the results
        evaluation_file_path = evaluate_file(result_tagged_file_path, OUTPUT_DIR)
    
    # check if improvement is requested
    elif IMPROVEMENT == "naive":
        
        # apply improvement to the train file
        train_additional_feature_naive_file_path = apply_naive_improvement(train_additional_feature_file_path)
        
        # apply improvement to the test file
        test_additional_feature_naive_file_path = apply_naive_improvement(test_additional_feature_file_path)
        
        # generate lexicon and transducer
        lexicon_file_path, transducer_file_path = generate_lexicon_and_transducer(train_additional_feature_naive_file_path, HANDLE_UNK, OUTPUT_DIR)
        
        # compile transducer
        transducer_fst_file_path = compile_transducer(lexicon_file_path, transducer_file_path, OUTPUT_DIR)
        
        # generate concept sentences file
        concept_sentences_file_path = create_concept_sentences(train_additional_feature_naive_file_path, OUTPUT_DIR)
        
        # generate language model
        language_model_lm_file_path = create_language_model(lexicon_file_path, concept_sentences_file_path, NGRAM_SIZE, SMOOTHING, BACKOFF, BINS, WITTEN_BELL_K, DISCOUNT_D, OUTPUT_DIR)
        
        # tag the test file
        result_tagged_file_path = tag_file(test_additional_feature_naive_file_path, lexicon_file_path, transducer_fst_file_path, language_model_lm_file_path, OUTPUT_DIR)       
        
        # evaluate the results
        evaluation_file_path = evaluate_file(result_tagged_file_path, OUTPUT_DIR)
    
    else: # IMPROVEMENT == "none"
        
        # generate lexicon and transducer
        lexicon_file_path, transducer_file_path = generate_lexicon_and_transducer(train_additional_feature_file_path, HANDLE_UNK, OUTPUT_DIR)
        
        # compile transducer
        transducer_fst_file_path = compile_transducer(lexicon_file_path, transducer_file_path, OUTPUT_DIR)
        
        # generate concept sentences file
        concept_sentences_file_path = create_concept_sentences(train_additional_feature_file_path, OUTPUT_DIR)
        
        # generate language model
        language_model_lm_file_path = create_language_model(lexicon_file_path, concept_sentences_file_path, NGRAM_SIZE, SMOOTHING, BACKOFF, BINS, WITTEN_BELL_K, DISCOUNT_D, OUTPUT_DIR)
        
        # tag the test file
        result_tagged_file_path = tag_file(test_additional_feature_file_path, lexicon_file_path, transducer_fst_file_path, language_model_lm_file_path, OUTPUT_DIR)       
        
        # evaluate the results
        evaluation_file_path = evaluate_file(result_tagged_file_path, OUTPUT_DIR)

else: # ADDITIONAL_FEATURE == "none"
    
    # check if improvement is requested
    if IMPROVEMENT == "wise":
        
        # apply improvement to the train file
        train_wise_file_path = apply_wise_improvement(train_temp_file_path)
        
        # apply improvement to the test file
        test_wise_file_path = apply_wise_improvement(test_temp_file_path)
        
        # generate lexicon and transducer
        lexicon_file_path, transducer_file_path = generate_lexicon_and_transducer(train_wise_file_path, HANDLE_UNK, OUTPUT_DIR)
        
        # compile transducer
        transducer_fst_file_path = compile_transducer(lexicon_file_path, transducer_file_path, OUTPUT_DIR)
        
        # generate concept sentences file
        concept_sentences_file_path = create_concept_sentences(train_wise_file_path, OUTPUT_DIR)
        
        # generate language model
        language_model_lm_file_path = create_language_model(lexicon_file_path, concept_sentences_file_path, NGRAM_SIZE, SMOOTHING, BACKOFF, BINS, WITTEN_BELL_K, DISCOUNT_D, OUTPUT_DIR)
        
        # tag the test file
        result_tagged_file_path = tag_file(test_wise_file_path, lexicon_file_path, transducer_fst_file_path, language_model_lm_file_path, OUTPUT_DIR)       
        
        # evaluate the results
        evaluation_file_path = evaluate_file(result_tagged_file_path, OUTPUT_DIR)
    
    # check if improvement is requested
    elif IMPROVEMENT == "naive":
        
        # apply improvement to the train file
        train_naive_file_path = apply_naive_improvement(train_temp_file_path)
        
        # apply improvement to the test file
        test_naive_file_path = apply_naive_improvement(test_temp_file_path)
        
        # generate lexicon and transducer
        lexicon_file_path, transducer_file_path = generate_lexicon_and_transducer(train_naive_file_path, HANDLE_UNK, OUTPUT_DIR)
        
        # compile transducer
        transducer_fst_file_path = compile_transducer(lexicon_file_path, transducer_file_path, OUTPUT_DIR)
        
        # generate concept sentences file
        concept_sentences_file_path = create_concept_sentences(train_naive_file_path, OUTPUT_DIR)
        
        # generate language model
        language_model_lm_file_path = create_language_model(lexicon_file_path, concept_sentences_file_path, NGRAM_SIZE, SMOOTHING, BACKOFF, BINS, WITTEN_BELL_K, DISCOUNT_D, OUTPUT_DIR)
        
        # tag the test file
        result_tagged_file_path = tag_file(test_naive_file_path, lexicon_file_path, transducer_fst_file_path, language_model_lm_file_path, OUTPUT_DIR)       
        
        # evaluate the results
        evaluation_file_path = evaluate_file(result_tagged_file_path, OUTPUT_DIR)
    
    else: # IMPROVEMENT == "none"
        
        # generate lexicon and transducer
        lexicon_file_path, transducer_file_path = generate_lexicon_and_transducer(train_temp_file_path, HANDLE_UNK, OUTPUT_DIR)
        
        # compile transducer
        transducer_fst_file_path = compile_transducer(lexicon_file_path, transducer_file_path, OUTPUT_DIR)
        
        # generate concept sentences file
        concept_sentences_file_path = create_concept_sentences(train_temp_file_path, OUTPUT_DIR)
        
        # generate language model
        language_model_lm_file_path = create_language_model(lexicon_file_path, concept_sentences_file_path, NGRAM_SIZE, SMOOTHING, BACKOFF, BINS, WITTEN_BELL_K, DISCOUNT_D, OUTPUT_DIR)
        
        # tag the test file
        result_tagged_file_path = tag_file(test_temp_file_path, lexicon_file_path, transducer_fst_file_path, language_model_lm_file_path, OUTPUT_DIR)       
        
        # evaluate the results
        evaluation_file_path = evaluate_file(result_tagged_file_path, OUTPUT_DIR)

