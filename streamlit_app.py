import streamlit as st
import numpy as np 
import pandas as pd

import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score,mean_absolute_error

# Input data & split
df = pd.read_csv('databn.csv')

X = df[['age',"sex",'Филиал','БН','grade','years_worked']].values
y = df['Left']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7,stratify=y) 

X2 = df[['age',"sex",'Филиал','БН','grade','Left']].values
y2 = df['years_worked']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.25, random_state=7)

# Model training

model = RandomForestClassifier(random_state=7)
model.fit(X_train, y_train)

model2 = RandomForestRegressor(random_state=0)
model2.fit(X_train2, y_train2)


bs = np.empty((0,0))
bs2 = np.empty((0,0))

st.set_page_config(page_title="Neoflex", page_icon="https://yt3.ggpht.com/ytc/AKedOLS-oo4mFL7rf95RUTFHgWjtAER5R5cnneqRGeTK=s900-c-k-c0x00ffffff-no-rj", layout="wide")
st.title("Система прогнозирования текучести кадров")

form = st.sidebar.form(key="input_form")
with form:

    # AGE

    age= st.number_input('Введите возраст сотрудника:', 17, max_value=70)
    age= np.array(age)

    bs = np.append(bs,age)
    bs2 = np.append(bs2,age)

    # SEX

    
    #st.write("Выберите пол сотрудника:")
    sex = st.radio("Выберите пол сотрудника:", ('Мужской', 'Женский'))
    if sex == 'Мужской':
        sex = 1
    else:
        sex = 0
    sex=int(sex)
    bs = np.append(bs,sex)
    bs2 = np.append(bs2,sex)

    # CITY

    
    city = st.selectbox("В каком филиале работает сотрудник?",['Москва','Саратов','Воронеж', 'Йоханнесбург', 'Пенза'])
    city_list = []
    city_list.append(city)
    city_pd = pd.DataFrame(city_list)
    city_n = city_pd.replace(['Москва','Саратов','Воронеж', 'Йоханнесбург', 'Пенза'],['2','4','0','1','3'])
    bs = np.append(bs,city_n)
    bs2 = np.append(bs2,city_n)
    
    # Бизнес-направление

    
    bn = st.selectbox("Бизнес-направление сотрудника:", df['BN'].unique())
    bn_list = []
    bn_list.append(bn)
    bn_pd = pd.DataFrame(bn_list)
    global BN_name
    global BN_number
    BN_name = df["BN"].sort_values().unique()
    BN_number = df["БН"].sort_values().unique()
    bn_n = bn_pd.replace(BN_name,BN_number)

    bs = np.append(bs,bn_n)
    bs2 = np.append(bs2,bn_n)

    # Грейд

    #grade = st.slider('Выберите грейд сотрудника:', -1,9)
    grade = st.select_slider('Выберите грейд сотрудника: ',options=['-1', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    bs = np.append(bs,grade)
    bs2 = np.append(bs2,grade)

    # Years worked

    years = st.number_input("Введите количество лет, проработанных сотрудником:", 0.0, max_value=13.0,step=0.5,format="%.1f")
    bs = np.append(bs,years)

    bs = bs.reshape(1,-1)

    bs2 = np.append(bs2,0)
    bs2 = bs2.reshape(1,-1)

    # Output
    prediction = np.array(model.predict_proba(bs))
    prediction = round(float(prediction[:,1])*100,2)
    out = "Вероятность ухода сотрудника:  **{}%**".format(prediction)

    prediction2 = np.array(model2.predict(bs2))
    prediction2 = round(float(prediction2),1)
    out2 = "Спрогнозированное количество лет, сколько сотрудник проработает: **{}**".format(prediction2)

    submitted = st.form_submit_button(label="Раcсчитать")


if submitted:
     #st.success("Спасибо!")

     def string(sex):
        sexstr = ""
        if sex == 0:
            sexstr = 'Женский'
        else:
            sexstr = 'Мужской'
        return sexstr

     def addyears(years):
        yearstr = ""
        if years == 1:
            yearstr = 'год'
        elif years > 1 and years < 5:
            yearstr = 'года'
        else:
            yearstr = 'лет'
        return yearstr

     st.write('Для сотрудника с входными параметрами:')
     st.write('Возраст:   ', str(age), '  \n Пол: ', string(sex), '  \n Город: ', city, 
        '  \n БН: ', bn, '  \n Грейд: ', str(grade), '  \n Стаж: ', str(years),addyears(years))
     st.write(out)
     st.write(out2)

     if prediction2 < 1:
        st.write("Сотрудник имеет **высокий** уровень риска текучести!")
     elif prediction2 >= 1 and prediction2 <=3:
        st.write("Сотрудник имеет **средний** уровень риска текучести")
     else:
        st.write("Сотрудник имеет **низкий** уровень риска текучести :thumbsup: ")
     

     fig = plt.figure(figsize=(1,1))
     ax = fig.subplots()
     sns.barplot(x=df['sex'], y=df['age'], color='goldenrod', ax=ax)
     ax.set_xlabel('sex')
     ax.set_ylabel('age')
     ax.set_title('Simple plot')
     #st.pyplot(fig)


#---------------------------------------------------------------
bs1 = pd.DataFrame(bs)
bs1.columns =['Возраст', 'Пол','Город','БН','Грейд','Стаж(лет)']
bs1['Пол'] = int(bs1['Пол'])
bs1['Пол'] = bs1['Пол'].replace([0,1],['Женский','Мужской'])
bs1['Город'] = bs1['Город'].replace(['2','4','0','1','3'],['Москва','Саратов','Воронеж', 'Йоханнесбург', 'Пенза'])
bs1['БН'] = bs1['БН'].replace(BN_number,BN_name)
bs1['Возраст'] = int(bs1['Возраст'])

print("R2: ", r2_score(y_test2, model2.predict(X_test2)),
      "\nMAE:",mean_absolute_error(y_test2, model2.predict(X_test2)), 
      "\nMSE:",np.sqrt(mean_squared_error(y_test2, model2.predict(X_test2))))
