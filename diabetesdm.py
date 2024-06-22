#!/usr/bin/env python
# coding: utf-8

# # Exploração de Dados

# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from yellowbrick.classifier import ClassificationReport
from yellowbrick.classifier import ClassPredictionError
from yellowbrick.classifier import ConfusionMatrix
from yellowbrick.classifier import ROCAUC
from yellowbrick.classifier import PrecisionRecallCurve

from lmfit.models import GaussianModel

import pickle

from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Para os graficos de relevancia de features

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import os #filesytem

#NN
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.metrics import f1_score, recall_score, precision_score

#usado em PCA
from sklearn.decomposition import PCA
from sklearn.metrics import  accuracy_score, classification_report

#usado em LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[3]:


#Classes auxiliares

class SBS:
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = estimator
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state
        
    def fit(self, X, y):
        dim = X.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X, y, self.indices_)
        
        while dim > self.k_features:
            scores = []
            subsets = []
            
            for p in combinations(self.indices_, r=dim - 1):
                score = self._calc_score(X, y, p)
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            
        self.k_score_ = self._calc_score(X, y, self.indices_)
        return self
    
    def transform(self, X):
        return X[:, self.indices_]
    
    def _calc_score(self, X, y, indices):
        self.estimator.fit(X[:, indices], y)
        y_pred = self.estimator.predict(X[:, indices])
        score = self.scoring(y, y_pred)
        return score


# In[4]:


# Funcoes auxiliares


# Funçoes para vizualizar em grafico os relatorios
# Classificacao Report

def Class_report(model,classes, X_train,X_test, y_train, y_test):
    visualizer = ClassificationReport(model, classes=classes, support=True)
    visualizer.fit(X_train, y_train)  # Ajustar o visualizador e o modelo
    visualizer.score(X_test, y_test)  # Avaliar o modelo nos dados de teste
    return visualizer.poof()
    
        
# Matriz de confusao  
def CM(model,classes, X_train, X_test, y_train, y_test ):
    visualizer = ConfusionMatrix(model, classes=classes,percent=False)
    visualizer.fit(X_train, y_train)  
    visualizer.score(X_test, y_test)  
    return visualizer.poof()


#Funcoes para gravacao/carregamento de modelos
# Gravacao Modelo
def ModelSave(model, filename):
    folder = 'Model'
    if not os.path.exists(folder):
        os.makedirs(folder)
    pickle_file = os.path.join(folder, filename) 
    with open(pickle_file, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {pickle_file}")

    
# Carregamento do modelo
def ModelLoad( filename):
    folder = 'Model' 
    pickle_file = os.path.join(folder, filename) 
    with open(pickle_file, 'rb') as file:
        loaded_model = pickle.load(file)
    print("Model loaded from disk")
    return loaded_model

    
#Funcoes auxiliares para redes neuronais   
def NNCF(y_test, y_pred):
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Imprimir a matriz de confusão para verificar sua estrutura
    print("Confusion Matrix:\n", conf_matrix)
    
    # Plot da matriz de confusão
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()    


# In[5]:


# Importar os dados
df = pd.read_csv("Data/Diabetes.csv")
print(df)
print(df.info())
print(df.describe().T)


# In[6]:


#Histograma - Distribuição dos Dados Iniciais
p = df.hist(figsize = (20,15))
    
plt.suptitle("Distribuição das Variáveis", y=1.02, fontsize=30)
plt.show()


# In[7]:


#Análise de Correlação de variáveis

correlations= df[:].corr()
mask = np.array(correlations)
mask[np.tril_indices_from(mask)] = False

plt.subplots(figsize=(15,12))
sns.heatmap(data=correlations,mask=mask, annot=True, fmt = '.2f', linewidths=1, linecolor="white",cmap = "RdBu_r", center=0, square=True);
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.title('Correlação entre as Variáveis', fontsize=19)
plt.show()


# In[8]:


#Análise de existência de elementos nulos
print(df.isnull().sum())


# # Preparação de Dados

# In[7]:


# Validar os tipos de dados

# Converter tipos de dados
df['Age'] = df['Age'].astype(int)
df['HighBP'] = df['HighBP'].astype(bool)
df['HighChol'] = df['HighChol'].astype(bool)  
df['CholCheck'] = df['CholCheck'].astype(bool)
df['Stroke'] = df['Stroke'].astype(bool)
df['HeartDiseaseorAttack'] = df['HeartDiseaseorAttack'].astype(bool)
df['PhysActivity'] = df['PhysActivity'].astype(bool)
df['Fruits'] = df['Fruits'].astype(bool)
df['Veggies'] = df['Veggies'].astype(bool)
df['HvyAlcoholConsump'] = df['HvyAlcoholConsump'].astype(bool)
df['AnyHealthcare'] = df['AnyHealthcare'].astype(bool)
df['NoDocbcCost'] = df['NoDocbcCost'].astype(bool)
df['DiffWalk'] = df['DiffWalk'].astype(bool)
df['Sex'] = df['Sex'].astype(bool)  
df['GenHlth'] = df['GenHlth'].astype(int)
df['MentHlth'] = df['MentHlth'].astype(int)
df['Education'] = df['Education'].astype(int)
df['Income'] = df['Income'].astype(int)
df['PhysHlth'] = df['PhysHlth'].astype(int)
df['Smoker'] = df['Smoker'].astype(bool)                  
df['Diabetes_binary'] = df['Diabetes_binary'].astype(bool)


# In[8]:


#Gravaçao de dados optimizados relativamente ao seu tipo
df.to_csv('Data/Diabetes_Optimized.csv', index=False)
print(df.info())


# In[9]:


#Histograma - Distribuição dos Dados com tipo validado
p = df.hist(figsize = (20,11))
    
plt.suptitle("Distribuição das Variáveis", y=1.02, fontsize=30)
plt.show()


# In[10]:


#Exploração/visualização estatistica 
#Facilita a interpretação 

def Diabetes_binary (row):
    
    if row['Diabetes_binary'] == 0:
        return 'Não'
    elif row['Diabetes_binary'] == 1:
        return 'Sim'
    
    
def HighBP (row):
    
     if row['HighBP'] == 0:
        return 'Não'
     elif row['HighBP'] == 1:
        return 'Sim'
    

def HighChol (row):
    
     if row['HighChol'] == 0:
        return 'Não'
     elif row['HighChol'] == 1:
        return 'Sim'  
    
    
def CholCheck (row):
    
     if row['CholCheck'] == 0:
        return 'Não'
     elif row['CholCheck'] == 1:
        return 'Sim'     
    
def Smoker (row):
    
     if row['Smoker'] == 0:
        return 'Não'
     elif row['Smoker'] == 1:
        return 'Sim' 

    
def Stroke (row):
    
     if row['Stroke'] == 0:
        return 'Não'
     elif row['Stroke'] == 1:
        return 'Sim' 
    
def HeartDiseaseorAttack (row):
    
     if row['HeartDiseaseorAttack'] == 0:
        return 'Não'
     elif row['HeartDiseaseorAttack'] == 1:
        return 'Sim' 
    
def PhysActivity (row):
    
     if row['PhysActivity'] == 0:
        return 'Não'
     elif row['PhysActivity'] == 1:
        return 'Sim' 
    
def Fruits (row):
    
     if row['Fruits'] == 0:
        return 'Não'
     elif row['Fruits'] == 1:
        return 'Sim' 

    
def Veggies (row):
    
     if row['Veggies'] == 0:
        return 'Não'
     elif row['Veggies'] == 1:
        return 'Sim' 

    
def HvyAlcoholConsump (row):
    
     if row['HvyAlcoholConsump'] == 0:
        return 'Não'
     elif row['HvyAlcoholConsump'] == 1:
        return 'Sim' 
    
def AnyHealthcare (row):
    
     if row['AnyHealthcare'] == 0:
        return 'Não'
     elif row['AnyHealthcare'] == 1:
        return 'Sim'

    
def DiffWalk (row):
    
     if row['DiffWalk'] == 0:
        return 'Não'
     elif row['DiffWalk'] == 1:
        return 'Sim'
    
    
def Sex (row):
    
     if row['Sex'] == 0:
        return 'Feminino'
     elif row['Sex'] == 1:
        return 'Masculino'

    
def NoDocbcCost (row):
    
    if row['NoDocbcCost'] == 0:
        return 'Não'
    elif row['NoDocbcCost'] == 1:
        return 'Sim' 
    
def GenHlth (row):
    
    if row['GenHlth'] == 1:
        return 'Excelente'
    elif row['GenHlth'] == 2:
        return 'Muito Boa'
    elif row['GenHlth'] == 3:
        return "Boa"
    elif row['GenHlth'] == 4:
        return "Má"
    elif row['GenHlth'] == 5:
        return "Péssima"

def Age (row):
    
    if row['Age'] == 1:
        return '18 à 24'
    
    elif row['Age'] == 2:
        return '25 à 29'
    
    elif row['Age'] == 3:
        return "30 à 34"
    
    elif row['Age'] == 4:
        return "35 à 39"
    
    elif row['Age'] == 5:
        return "40 à 44"
    
    elif row['Age'] == 6:
        return "45 à 49"
    
    elif row['Age'] == 7:
        return "50 à 54"
    
    elif row['Age'] == 8:
        return "55 à 59"
    
    elif row['Age'] == 9:
        return "60 à 64"
    
    elif row['Age'] == 10:
        return "65 à 69"
    
    elif row['Age'] == 11:
        return "70 à 74"
    
    elif row['Age'] == 12:
        return "75 à 79"
    
    elif row['Age'] == 13:
        return "80+"
    

def Education (row):
    
    if row['Education'] == 1:
        return 'Nunca frequentou a escola ou apenas o jardim de infância'
    elif row['Education'] == 2:
        return 'Ensino Básico'
    elif row['Education'] == 3:
        return "Ensino Secundário"
    elif row['Education'] == 4:
        return "Concluiu o ensino secundário"
    elif row['Education'] == 5:
        return "Alguma faculdade ou escola técnica"
    elif row['Education'] == 6:
        return "Licenciatura"
    
def Income (row):
 
    if row['Income'] == 1:
        return '<10000$'
    elif row['Income'] == 2:
        return '10000$ à 15000$'
    elif row['Income'] == 3:
        return "15000$ à 20000$"
    elif row['Income'] == 4:
        return "20000$ à 25000$"
    elif row['Income'] == 5:
        return "25000$ à 35000$"
    elif row['Income'] == 6:
        return "35000$ à 50000$"
    elif row['Income'] == 7:
        return "50000$ à 75000$"
    elif row['Income'] == 8:
        return ">75000$"


# In[11]:


#Construçao de DataFrame para uma mais fácil visualização dos dados

dn = pd.DataFrame()

dn['Diabetes_binary'] = df.apply(Diabetes_binary, axis = 1)  
dn['HighBP'] = df.apply(HighBP, axis = 1)  
dn['HighChol'] = df.apply(HighChol, axis = 1)  
dn['CholCheck'] = df.apply(CholCheck, axis = 1)
dn['Smoker'] = df.apply(Smoker, axis = 1)
dn['Stroke'] = df.apply(Stroke, axis = 1)
dn['HeartDiseaseorAttack'] = df.apply(HeartDiseaseorAttack, axis = 1)
dn['PhysActivity'] = df.apply(PhysActivity, axis = 1)
dn['Fruits'] = df.apply(Fruits, axis = 1)
dn['Veggies'] = df.apply(Veggies, axis = 1)
dn['HvyAlcoholConsump'] = df.apply(HvyAlcoholConsump, axis = 1)
dn['AnyHealthcare'] = df.apply(AnyHealthcare, axis = 1)
dn['DiffWalk'] = df.apply(DiffWalk, axis = 1)
dn['Sex'] = df.apply(Sex, axis = 1)
dn['NoDocbcCost'] = df.apply(NoDocbcCost, axis = 1)
dn['GenHlth'] = df.apply(GenHlth, axis = 1)
dn['Age'] = df.apply(Age, axis = 1)
dn['Education'] = df.apply(Education, axis = 1)
dn['Income'] = df.apply(Income, axis = 1)
dn['BMI'] = df['BMI'] 
dn['MentHlth'] = df['MentHlth'] 
dn['PhysHlth'] = df['PhysHlth'] 

#Com a conversão feita anteriormente, a leitura dos dados torna-se mais fácil, principalmente para as variáveis ordinais
print(dn.describe(include=['O']) )
print(dn.head().T)


# In[12]:


#Análise de outliers em variaveis numéricas

colnames = df[['BMI', 'MentHlth', 'PhysHlth']].columns

fig, axs = plt.subplots(3, 1, figsize=(10, 5))
fig.suptitle('Boxplot das Variáveis BMI, MentHlth e PhysHlth', fontsize=16) 

for i, col in enumerate(colnames):
    sns.boxplot(x=df[col], data=df, ax=axs[i], width=0.5, linewidth=1, fliersize=3)
    axs[i].set_title(col)  

plt.tight_layout()
plt.show()


# In[13]:


# Skewness deve estar entre -1 a 1 para uma distribuição normal
print('skewness value of BMI: ',df['BMI'].skew())
print('skewness value of MentHlth: ',df['MentHlth'].skew())
print('skewness value of PhysHlth: ',df['PhysHlth'].skew())

#os Valores não apresentam uma distribuição normal


# In[14]:


#Distribuição da variavel BMI, uma das que foi do nosso entendimento ser mais relevante

plt.figure(figsize=(10, 4))
sns.histplot(data=dn, x='BMI', kde=True, color = 'blue')
plt.title('Distribuição do BMI')
plt.xlabel('BMI')
plt.ylabel('Contagem')
plt.show()


# In[15]:


#Distribuiçao das variaveis representativas de idade e estado geral de saúde

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

sns.countplot(x='Age', data=dn, ax=axes[0], color = 'blue')
axes[0].set_xlabel('Idade Agrupada')
axes[0].set_ylabel('Contagem')
axes[0].set_title('Idade', fontsize=16)

sns.histplot(x='GenHlth', data=dn, ax=axes[1])
axes[1].set_xlabel('Saúde Geral')
axes[1].set_ylabel('Contatem')
axes[1].set_title('Condição Geral de Saúde', fontsize=16)

plt.tight_layout()
plt.suptitle('', fontsize=16)
plt.show()


# In[16]:


#Distribuiçao das variaveis representativas de nível de educaçao e rendimento anual
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

dn['Education'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
ax1.set_title('Nível de Educação', fontsize=16)
ax1.set_ylabel('')
ax1.axis('equal') 

dn['Income'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
ax2.set_title('Rendimento Anual em Classes', fontsize=16)
ax2.set_ylabel('')
ax2.axis('equal') 

plt.tight_layout()

plt.show()


# In[17]:


#continuação de exploração de variáveis

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

dn['AnyHealthcare'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
ax1.set_title('Seguro de saúde', fontsize=12)
ax1.set_ylabel('')
ax1.axis('equal') 

dn['NoDocbcCost'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
ax2.set_title('Impossibilidade de ir ao médico devido aos custos', fontsize=12)
ax2.set_ylabel('')
ax2.axis('equal')  

plt.tight_layout()

plt.show()


# In[18]:


#análise por género
fig = plt.figure(figsize=(4,4))

sns.countplot(x='Sex', data=dn)

plt.xlabel('Género')
plt.ylabel('Contagem')

plt.suptitle('Género', fontsize=10)


# In[19]:


#continuação de exploração de variáveis (saúde mental)
fig = plt.figure(figsize=(10,4))

sns.countplot(x='MentHlth', data=dn)

plt.xlabel('Saúde Mental')
plt.ylabel('Contagem')

plt.suptitle('Dias em que se sentiu mal psicologicamente', fontsize=14)


# In[20]:


#continuação de exploração de variáveis Problemas Físicos/Lesões
fig = plt.figure(figsize=(10,4))

sns.countplot(x='PhysHlth', data=dn)

plt.xlabel('Problemas Físicos/Lesões')
plt.ylabel('Contagem')

plt.suptitle('Dias em que teve problemas físicos ou lesões', fontsize=14)


# In[21]:





# In[22]:


#Distribuição da variavel alvo "Tem Diabetes/Não tem Diabetes"
sns.catplot(data=df, x="Diabetes_binary", kind='count', palette='rocket')
plt.suptitle('Distribuição da Variável Alvo', fontsize=14)


# ## Seleção de Modelos, Seleção de Features e Redução de Dimensionalidade

# In[23]:


#S Seleção de Features e Redução de Dimensionalidade
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, shuffle=True, test_size=0.15)


# In[24]:


# SMOTE implmentação da técnica SMOTE para equilibrar resultados
smote = SMOTE()

X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# In[25]:


#Balanceamento do dataset
df_balanced=X_resampled
df_balanced['Diabetes_binary']=y_resampled

print(df_balanced.info())


# In[26]:


#Dados Balanceados
sns.catplot(data=df_balanced, x="Diabetes_binary", kind='count', palette='rocket')
plt.suptitle('Distribuição da Variável Alvo', fontsize=14)


# In[27]:


#Construção do Dataset Final
dfFinal=df_balanced

dfFinal.to_csv('Data/Diabetes_Final.csv', index=False)

print(dfFinal.head())


# In[28]:


#Correlação - Dados Balanceados 
correlations= dfFinal[:].corr()
mask = np.array(correlations)
mask[np.tril_indices_from(mask)] = False

plt.subplots(figsize=(15,12))
sns.heatmap(data=correlations,mask=mask, annot=True, fmt = '.2f', linewidths=1, linecolor="white",cmap = "RdBu_r", center=0, square=True);
b, t = plt.ylim()
b += 0.5
t -= 0.5
plt.ylim(b, t)
plt.title('Correlação entre as Variáveis\nDados Balanceados', fontsize=16)
plt.show()


# # Classificação

# In[29]:


# Remover variaveis com pouca coorelação com a nossa variavel alvo
dfFinal = dfFinal.drop(['Fruits', 'Veggies','AnyHealthcare','NoDocbcCost','MentHlth','Sex'], axis=1)

dfFinal.to_csv('Data/Diabetes_Final.csv', index=False)


# In[30]:


X = dfFinal.drop(columns=['Diabetes_binary'])
y = dfFinal['Diabetes_binary']

classes = [0, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)


# In[31]:


#Normalização de features
sc=StandardScaler()
X_train =sc.fit_transform(X_train)
X_test =sc.transform(X_test)


# In[46]:


# Regressão Logística com Feature Selection

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(y_train)

print("Logistic Regression Accuracy: ", np.round(accuracy_score(y_test, y_pred)*100, 2), "%")

Class_report(model, classes, X_train, X_test, y_train, y_test)
CM(model, classes, X_train, X_test, y_train, y_test)

ModelSave(model, 'LogisticRegression')


# In[47]:


# K Nearest Neighbours com Feature Selection

model = KNeighborsClassifier(n_neighbors = 3)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("K Nearest Neighbours: ", np.round(accuracy_score(y_test, y_pred)*100, 2), "%")

Class_report(model, classes, X_train, X_test, y_train, y_test)
CM(model, classes, X_train, X_test, y_train, y_test)

ModelSave(model, 'KNeighborsClassifier')


# In[ ]:


# Support Vector Machines com Feature Selection

model = SVC()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RSupport Vector Machines Accuracy: ", np.round(accuracy_score(y_test, y_pred)*100, 2), "%")

Class_report(model, classes), X_train, X_test, y_train, y_test
CM(model, classes, X_train, X_test, y_train, y_test)

ModelSave(model, 'SVC')


# In[ ]:


# Naives Bayes

model = GaussianNB()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RSupport Vector Machines Accuracy: ", np.round(accuracy_score(y_test, y_pred)*100, 2), "%")

Class_report(model, classes, X_train, X_test, y_train, y_test)
CM(model, classes, X_train, X_test, y_train, y_test)

ModelSave(model, 'GaussianNB')


# In[ ]:


# DecisionTree

model = DecisionTreeClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("DecisionTree Accuracy: ", np.round(accuracy_score(y_test, y_pred)*100, 2), "%")

Class_report(model, classes, X_train, X_test, y_train, y_test)
CM(model, classes, X_train, X_test, y_train, y_test)

ModelSave(model, 'DecisionTreeClassifier')


# In[ ]:


# Random forests 

model = RandomForestClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Random Forest Accuracy: ", np.round(accuracy_score(y_test, y_pred)*100, 2), "%")

Class_report(model, classes, X_train, X_test, y_train, y_test)
CM(model, classes, X_train, X_test, y_train, y_test)

ModelSave(model, 'RandomForestClassifier')


# In[74]:


#Analise visual da relevancia das features no mdelo de Random Forest
dfFeatures = pd.get_dummies(df)

X = dfFeatures.drop('Diabetes_binary', axis=1)
y = dfFeatures['Diabetes_binary']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
sns.barplot(x=importances[indices], y=X.columns[indices], orient='h')
plt.show()


# #### Redução de Features
# 

# In[72]:


# Sequential Backward Selection (SBS)

from itertools import combinations
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression()
model.fit(X_train, y_train)

# calculo do modelo com 5 features
sbs = SBS(estimator=LogisticRegression(), k_features=5)
sbs.fit(X_train, y_train)

print(sbs.indices_)

feature_names = dfFinal.columns[:-1] 
selected_features_names = [feature_names[i] for i in sbs.indices_]
print("Features selecionadas:")
print(selected_features_names)

#Teste do modelo
X_train_sbs = sbs.transform(X_train)
X_test_sbs = sbs.transform(X_test)

model.fit(X_train_sbs, y_train)
print('Performance do modelo com características selecionadas:')
print(accuracy_score(y_test, model.predict(X_test_sbs)))


# In[73]:


#SFS SequentialFeatureSelector
from sklearn.feature_selection import SequentialFeatureSelector 

# Definindo o classificador
classifierSFS = RandomForestClassifier(n_estimators=100, random_state=42)

sfs = SequentialFeatureSelector(classifierSFS, n_features_to_select=5, direction='forward')  # Ajuste o número de features

sfs.fit(X_train, y_train)

selected_features = sfs.get_support(indices=True)
print("Selected features indices:", selected_features)

X_train_sfs = sfs.transform(X_train)
X_test_sfs = sfs.transform(X_test)

classifierSFS.fit(X_train_sfs, y_train)

# Avaliação do modelo
y_pred = classifierSFS.predict(X_test_sfs)
print("Accuracy on test set with selected features:", accuracy_score(y_test, y_pred))


# #### Redução de Dimensionalidade
# 

# In[67]:


#PCA 
#Qualquer que seja o número de variáveis independentes no nosso 
#problema, podemos reduzir a duas variáveis independentes 
#utilizando a técnica de redução de dimensionalidade mais correta
#Principal Component Analysis (PCA)
#▻ A partir das m variáveis independentes do conjunto de dados 
#(a.k.a. dataset), PCA extrai p<m novas variáveis independentes 
#que explicam da melhor forma possível a variância do conjunto 
#de dados independentemente da variável dependente
#▻ Reduzir as dimensões de um dataset com d dimensões através 
#da projeção num sub-espaço de dimensão k, onde k<d


pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Explained variance ratio:")
print(pca.explained_variance_ratio_)

feature_names = dfFinal.columns[:-1]

pca_components = pd.DataFrame(pca.components_, columns=feature_names, index=['PC1', 'PC2'])

print("\nPrincipal Components (loading scores):")
print(pca_components)

Class_report(model, classes, X_train_pca, X_test_pca, y_train, y_test)
CM(model, classes, X_train_pca, X_test_pca, y_train, y_test)

ModelSave(model, 'LogisticRegression_WPCA')


plt.figure(figsize=(14, 6))
plt.bar(pca_components.columns, pca_components.loc['PC1'], alpha=0.5, align='center', label='PC1')
plt.bar(pca_components.columns, pca_components.loc['PC2'], alpha=0.5, align='center', label='PC2')
plt.xlabel('Features')
plt.ylabel('Contribution to Principal Components')
plt.title('PCA Component Loading Scores')
plt.xticks(rotation=90)
plt.legend()
plt.show()

model = LogisticRegression(random_state=0)
model.fit(X_train_pca, y_train)
#print(X_train_pca.shape)
#print(X_test_pca.shape)

y_pred = model.predict(X_test_pca)

# Confusion Matrix and Accuracy Score
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

plt.figure(figsize=(10, 7))

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolor='k', cmap=plt.cm.coolwarm, alpha=0.5)

plt.title('PCA of Diabetes Data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()


# #### Aplicacao LDA

# In[65]:


# LDA Linear Discriminant Analysis (LDA)

feature_names = dfFinal.columns[:-1]
lda = LDA(n_components=1)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

print("\nLDA Scalings (coefficients):")
print(lda.scalings_)

lda_coefficients = pd.DataFrame(lda.scalings_, index=feature_names, columns=['LD1'])

print("\nLDA Coefficients:")
print(lda_coefficients)

model = LogisticRegression()
model.fit(X_train_lda, y_train)

Class_report(model, classes, X_train_lda, X_test_lda, y_train, y_test)
CM(model, classes, X_train_lda, X_test_lda, y_train, y_test)

ModelSave(model, 'LogisticRegression_WLDA')

y_pred = model.predict(X_test_lda)

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))

class_names = ["No Diabetes", "Diabetes"]
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))

plt.figure(figsize=(10, 7))

sns.histplot(X_train_lda[y_train == 0], kde=True, color='blue', label='No Diabetes', stat="density", linewidth=0)
sns.histplot(X_train_lda[y_train == 1], kde=True, color='red', label='Diabetes', stat="density", linewidth=0)

plt.title('LDA of Diabetes Data')
plt.xlabel('LD1')
plt.legend()
plt.show()


# #### Associação

# In[ ]:


dataset = pd.read_csv('Data/Diabetes_Final.csv',  low_memory=False)

dataset = DataSetTypeCast(dataset)

dataset.head()

dataset.info()


transactions = []


for i in range(0, 7501):
  transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
  
transactions

from apyori import apriori

rules = apriori(transactions = transactions, min_support = 0.002, min_confidence = 0.2, min_lift = 2, min_length = 2, max_length = 5)


results = list(rules)

print("Número de regras encontradas:",len(results))

 

print("Primeira regra encontrada:",results[0])



results


for item in results:
    pair = item[0] 
    items = [x for x in pair]
    print("Regra: " + items[0] + " -> " + items[1])

    print("Support: " + str(item[1]))
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
    
    
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])


resultsinDataFrame


resultsinDataFrame.nlargest(n = 10, columns = 'Support')

