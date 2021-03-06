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


# importing a custom dataset which must be a pandas.dataframe or csv file
data = pd.read_csv('data/adult_cleared.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)

# generate dinamically names for syntethic dataset (e.g. "synthetic_adult_1" , "synthetic_adult_2")
pathToSave = getSavePath("Synthetic_data", "synthetic_adult")

# number of samples that we desire to generate
num_samples = 400

# model save location
model_path = 'models/Adult_2.pkl'

# the TGAN model need to know which dataset columns are the type of continuous columns. 
continuous_columns = [0, 2, 4, 10, 11, 12]

# set nn parameters, like epoch, batch size and loss function
tgan = TGANModel(continuous_columns,
                  output='output',
                  max_epoch=10,
                  steps_per_epoch=400,
                  save_checkpoints=True,
                  restore_session=False,
                  batch_size=200,
                  z_dim=200,
                  noise=0.2,
                  l2norm=0.00001,
                  learning_rate=0.001,
                  num_gen_rnn=100,
                  num_gen_feature=100,
                  num_dis_layers=1,
                  num_dis_hidden=100,
                  optimizer='AdamOptimizer')
                  
# train phase 
tgan.fit(data)

print("Fitted model!!!")

# after fitting, we can sample some new synthetic data which is a pandas.DataFrame
samples = tgan.sample(num_samples)

print(samples)

samples.head()

# save generated data as csv file and remove index line (first line) 
samples.to_csv(pathToSave, index=False)

# save the model. Use force = true to overwrite
tgan.save(model_path, force=True)