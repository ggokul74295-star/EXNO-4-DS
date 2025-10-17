# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("bmi.csv")
df.head()

<img width="335" height="232" alt="Screenshot 2025-10-17 193609" src="https://github.com/user-attachments/assets/011a2208-dff8-41a8-a461-65c1e2451cb7" />

df.dropna()

<img width="339" height="486" alt="Screenshot 2025-10-17 193622" src="https://github.com/user-attachments/assets/a98faf2f-8e38-4050-bf14-b263122000eb" />

df_null_sum=df.isnull().sum()
df_null_sum

<img width="149" height="135" alt="Screenshot 2025-10-17 193630" src="https://github.com/user-attachments/assets/4c670d52-5446-49ee-86d4-6bfbdc542574" />

max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
max_vals

<img width="178" height="81" alt="Screenshot 2025-10-17 193636" src="https://github.com/user-attachments/assets/c8a21e58-a08e-45c1-a19c-5f29d71a1fbf" />

from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()
from sklearn.preprocessing import StandardScaler
df1=pd.read_csv("bmi.csv")
df1.head()

<img width="332" height="250" alt="Screenshot 2025-10-17 193643" src="https://github.com/user-attachments/assets/15ed3557-2096-4bcb-a2b2-fa7defa71f75" />

df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

<img width="369" height="418" alt="Screenshot 2025-10-17 193648" src="https://github.com/user-attachments/assets/9a9fee50-f7f3-4863-8808-c2e6e208b5c3" />

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

<img width="362" height="486" alt="Screenshot 2025-10-17 193856" src="https://github.com/user-attachments/assets/831830c4-b93a-4e88-8c72-c4c3fa597510" />

from sklearn.preprocessing import MaxAbsScaler
scaler = MaxAbsScaler()
df3=pd.read_csv("bmi.csv")
df3.head()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df

<img width="374" height="236" alt="Screenshot 2025-10-17 193907" src="https://github.com/user-attachments/assets/c963e9af-34a0-4d5d-8b4f-3a8cf99eba9c" />

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3.head()

<img width="467" height="458" alt="Screenshot 2025-10-17 193915" src="https://github.com/user-attachments/assets/c8e82f0b-dac8-4878-8085-cb027e0d85ce" />

df=pd.read_csv("titanic_dataset.csv")
df.info()

<img width="267" height="322" alt="Screenshot 2025-10-17 193923" src="https://github.com/user-attachments/assets/4d794af2-18e8-427a-9ef9-2ef2c8cd3095" />

df_null_sum=df.isnull().sum()
df_null_sum

<img width="1020" height="381" alt="Screenshot 2025-10-17 193932" src="https://github.com/user-attachments/assets/731725d2-914a-41dc-a7ca-a3751252ed86" />

categorical_columns = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch','Ticket','Fare','Cabin','Embarked']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]

<img width="908" height="486" alt="Screenshot 2025-10-17 193939" src="https://github.com/user-attachments/assets/f0e5979d-88a0-4068-a0c9-7ba9a0329690" />

df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]

<img width="610" height="504" alt="Screenshot 2025-10-17 193951" src="https://github.com/user-attachments/assets/9ebf374a-fd73-458c-bafb-108206a5194f" />

X = df.drop(columns=['Survived'])
y = df['Survived']
df1=df.drop(columns=['Name','Sex','Ticket','Cabin','Embarked'])
df1['Age'].isnull().sum()

<img width="1011" height="34" alt="Screenshot 2025-10-17 194040" src="https://github.com/user-attachments/assets/eb802e7d-6930-49e8-a22e-9942353a525a" />

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns

<img width="925" height="39" alt="Screenshot 2025-10-17 194050" src="https://github.com/user-attachments/assets/18d4dc02-9543-4a2e-9f49-60e7b444661d" />

X=df1.iloc[:,0:6]
y=df1.iloc[:,6]
X.columns
<img width="507" height="203" alt="Screenshot 2025-10-17 194058" src="https://github.com/user-attachments/assets/f9091059-d5de-414b-be77-db5333bf6ee8" />
























































       
# RESULT:
      Thus, the program to implement Feature Scaling and Feature Selection was completed successfully.
