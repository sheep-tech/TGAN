'''
from tgan.data import load_demo_data

data, continuous_columns = load_demo_data('census')
print(data.head(3).T[:10])

print(continuous_columns)


from tgan.model import TGANModel
# to generate Tgan model, you need only the continuos columns
tgan = TGANModel(continuous_columns)
#The fit step takes few hours 
#tgan.fit(data)

# after fitting, we can sample some new synthetic data which is a pandas.DataFrame
num_samples = 1000
samples = tgan.sample(num_samples)
samples.head(3).T[:10]

# save the model. Use force= true overwrite
model_path = 'models/Census.pkl'
tgan.save(model_path)
'''
# importing a custom dataset which must be a pandas.dataframe or csv file
import pandas as pd
from tgan.model import TGANModel
data = pd.read_csv('data/adult_cleared.csv')
data.drop(['Unnamed: 0'], axis=1, inplace=True)
# continuous_columns = [0, 5, 16, 17, 18, 29, 38]
continuous_columns = [0, 2, 4, 10, 11, 12]
#print(data.head(3).T[:10])

#print(continuous_columns)
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
                  
#The fit step takes few hours 
tgan.fit(data)
# print("Fitted model!!!")
# # after fitting, we can sample some new synthetic data which is a pandas.DataFrame
# model_path = 'models/Census.pkl'

# tgan = TGANModel.load(model_path)

num_samples = 400
samples = tgan.sample(num_samples)
print(samples)
samples.head()

samples.to_csv("Synthetic_data/synthetic_adult_6.csv", index=False)
# # save the model. Use force= true overwrite
model_path = 'models/Adult_2.pkl'
tgan.save(model_path, force=True)