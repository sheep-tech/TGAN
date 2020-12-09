import matplotlib.pyplot as plt
import pandas as pd

original_data = pd.read_csv('data/adult.csv')
missing_header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain','capital-loss', 'hours per week', 'native-country', 'class']

adult = open("data/Dataset.data", "r")
countries = pd.read_csv('data/factbook.csv', delimiter=';', nrows=100)
adult = pd.read_csv(adult, delimiter=" ", header=None, names=missing_header)

# print(adult.iloc[:10, 12], adult.iloc[:10, 14])
# countries.head()
# plt.scatter(adult.iloc[:20, 12], adult.iloc[:20, 14])
# print(adult.iloc[:, 10].mean())
# print(adult.iloc[:, 10].std())
# print(adult['occupation'].median())
# new = adult.groupby(adult['occupation']).size().reset_index(name='counts')
# print(new.sort_values(new.columns[1], ascending=False))
# print(new.columns[1].median())


adult['workclass'].replace(to_replace='?', value="private", inplace=True)
adult['occupation'].replace(to_replace='?', value="Prof-specialty", inplace=True)
adult['native-country'].replace(to_replace='?', value="United-States", inplace=True)

# for x in missing_header:
#     print(adult[x].value_counts())
adult.to_csv("data/adult_cleared.csv")
# plt.bar(['<50k,', '>50k'], height=adult.iloc[:100, 14])
# plt.scatter(original_data["ID"], original_data["Education"])
# plt.xlabel("Area")
# plt.ylabel("Country")
# plt.show()
# #original_data.head()