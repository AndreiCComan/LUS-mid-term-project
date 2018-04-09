
# coding: utf-8

# ![DISI](../resources/DISI.jpeg)

# ### Import

# In[17]:


import json
import subprocess


# In[18]:


CONFIGURATION_FILE_NAME = "config.json"


# In[19]:


assert(subprocess.call("[[ -e config.json ]] && cp config.json config_bak.json", shell = True) == 0),"> unable to make a copy of configuration file"


# In[20]:


BASELINES = ["random", "chance", "majority"]


# In[21]:


try:
    for baseline in BASELINES:
        print("> running {0} baseline".format(baseline))
        
        configuration = {}
        configuration["baseline"] = baseline
        
        configuration_file = open(CONFIGURATION_FILE_NAME, 'w', encoding="utf8")
        json.dump(configuration, configuration_file)
        configuration_file.close()
        
        assert(subprocess.call("python3 baseline.py", shell = True) == 0),        "> unable to run the script"
except:
    pass


# In[22]:


assert(subprocess.call("[[ -e config_bak.json ]] && mv config_bak.json config.json", shell = True) == 0),"> unable to make a copy of configuration file"

