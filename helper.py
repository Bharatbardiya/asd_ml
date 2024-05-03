from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Commented out IPython magic to ensure Python compatibility.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# %matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# from keras.models import load_model
color = sns.color_palette()
sns.set_style('darkgrid')


from joblib import Parallel, delayed
import joblib


class ModelTrain:
    
    def loadData(self):
        data1=pd.read_csv('dataset/data_csv.csv')
        data2=pd.read_csv('dataset/Toddler Autism dataset July 2018.csv')
        data3=pd.read_csv('dataset/autism_screening.csv')
        
        self.df1=pd.concat([data1.iloc[:,1:11],data1.iloc[:,[12,22,23,24,25,26,27]]],axis=1)
        
        self.df2=pd.concat([data2.iloc[:,1:12],data2.iloc[:,13:]],axis=1)
        self.df2['Age_Mons']=(self.df2['Age_Mons']/12).astype(int)
        
        self.df3=pd.concat([data3.iloc[:,0:15],data3.iloc[:,-2:]],axis=1)
    
    def dataClening(self):
        
        self.loadData()
        # Rename columns to have the same names in all DataFrames
        self.df2.columns = self.df3.columns = self.df1.columns

        # Concatenate the DataFrames
        self.data_fin = pd.concat([self.df3, self.df2, self.df1], axis=0)
        
        # Get object type columns
        # object_cols = self.data_fin.select_dtypes('O').columns

        # Create new DataFrame
        # object_df = pd.DataFrame({
        #     'Objects': object_cols,
        #     'Unique values': [data_fin[col].unique() for col in object_cols],
        #     'number of unique values':[data_fin[col].nunique()for col in object_cols]
        # })

        # object_df

        self.data_fin.columns
        replacements = {
            'f': 'F',
            'm': 'M',
        }
        self.data_fin['Sex'] = self.data_fin['Sex'].replace(replacements)
        replacements = {
            'yes': 'Yes',
            'no': 'No',
        }
        self.data_fin['Jaundice'] = self.data_fin['Jaundice'].replace(replacements)
        replacements = {
            'yes': 'Yes',
            'no': 'No',
        }
        self.data_fin['Family_mem_with_ASD'] = self.data_fin['Family_mem_with_ASD'].replace(replacements)
        replacements = {
            'YES': 'Yes',
            'NO': 'No',
        }
        self.data_fin['ASD_traits'] = self.data_fin['ASD_traits'].replace(replacements)

        replacements = {
            'middle eastern': 'Middle Eastern',
            'Middle Eastern ': 'Middle Eastern',
            'mixed': 'Mixed',
            'asian': 'Asian',
            'black': 'Black',
            'south asian': 'South Asian',
            'PaciFica':'Pacifica',
            'Pasifika':'Pacifica'

        }
        self.data_fin['Ethnicity'] = self.data_fin['Ethnicity'].replace(replacements)

        replacements = {
            'Health care professional':'Health Care Professional',
            'family member':'Family Member',
            'Family member':'Family Member'
        }
        self.data_fin['Who_completed_the_test'] = self.data_fin['Who_completed_the_test'].replace(replacements)


        self.data_fin['Ethnicity'].replace('?', np.nan, inplace=True)
        self.data_fin['Who_completed_the_test'].replace('?', np.nan, inplace=True)
        pd.DataFrame(self.data_fin.isnull().sum(),
                    columns=["Missing Values"]).style.bar(color = "#84A9AC")
        
    def modelTrain(self):
        
        self.dataClening()
        
        idf = self.data_fin.copy()
        imp = SimpleImputer(strategy='most_frequent')
        imputed_data = pd.DataFrame(imp.fit_transform(idf))
        imputed_data.columns = idf.columns
        imputed_data.index = idf.index

        pd.DataFrame(imputed_data.isnull().sum(),
                    columns=["Missing Values"]).style.bar(color = "#84A9AC")

        data = imputed_data.copy()

        lr = LinearRegression()
        dtc = DecisionTreeClassifier()
        gclf1 =GaussianNB()
        mclf2 = MultinomialNB()
        bclf3 =  BernoulliNB()
        knn = KNeighborsClassifier()
        lgr = LogisticRegression()
        rfc = RandomForestClassifier(max_depth = 10, random_state=0)

        data = pd.get_dummies(data, columns= ['Ethnicity', 'Who_completed_the_test'], drop_first =  True)
        data.head()

        data.columns

        data['Sex'].replace({"M":1, "F":0}, inplace = True)
        data['Jaundice'].replace({"Yes":1, "No":0}, inplace = True)
        data['Family_mem_with_ASD'].replace({"Yes":1, "No":0}, inplace = True)
        # data['ASD_traits'].replace({"Yes":1, "No":0}, inplace = True)
        data.head()
        # print(data.columns)

        y = data['ASD_traits']
        x = data.drop(columns = ['ASD_traits'])
        print(x.head())
        print(y.head())
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(x, y, random_state=12, test_size=0.2)

        dtc.fit(X_train, Y_train)

        return dtc
