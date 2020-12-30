import pandas as pd
import os
from tgan.model import TGANModel

# generate dinamically different names to save new generated datasets
def getSavePath(path, name):
    files = os.listdir(path)
    index = 1

    for file in files:
        if(".csv" in file):
            try:
                number = int(file.split("_")[2][0:1])
            except:   
                number = 0
                
            if(number > index):
                index = number + 1

    return path + "/" + name + "_" + str(index) + ".csv"


# generate dinamically names for syntethic dataset (e.g. "synthetic_adult_1" , "synthetic_adult_2")
pathToSave = getSavePath("Synthetic_data", "synthetic_adult")

# number of samples that we desire to generate
num_samples = 400

# trained model location
model_path = 'models/Adult_2.pkl'

# load tgan model that was previously trained
tgan = TGANModel.load(model_path)

# after fitting, we can sample some new synthetic data which is a pandas.DataFrame
samples = tgan.sample(num_samples)

print(pathToSave)
print(samples)

samples.head()

# save generated data as csv file and remove index line (first line) 
samples.to_csv(pathToSave, index=False)

# save the model. Use force = true to overwrite
tgan.save(model_path, force=True)