import pickle
import time
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

Classifier=Sequential();
Classifier.add(Conv2D(64,(3,3),input_shape=(64,64,3),activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))
Classifier.add(Conv2D(32,(3,3),activation='relu'))
Classifier.add(MaxPooling2D(pool_size=(2,2)))
Classifier.add(Flatten())
Classifier.add(Dense(units=104, activation='relu'))
Classifier.add(Dense(units=1, activation='sigmoid'))
Classifier.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.4,
                                   zoom_range = 0.3,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory(r'E:\Data Analyst Journey\5. Streamlits Web Apps\9. JSPM BE Project\Covid19-dataset\train',
                                                 target_size = (64, 64),
                                                 batch_size = 4,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory(r'E:\Data Analyst Journey\5. Streamlits Web Apps\9. JSPM BE Project\Covid19-dataset\test',
                                            target_size = (64, 64),
                                            batch_size = 4,
                                            class_mode = 'binary')


st.set_page_config(page_title="Covid 19 Prediction", page_icon=":bar_chart:",layout="wide")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

data = pd.read_csv('covid.csv')

def split_data(df, ratio):
    mixed = np.random.permutation(len(df))
    test_size = int(len(df) * ratio)
    test_indices = mixed[:test_size]
    train_indices = mixed[test_size:]

    return df.iloc[train_indices], df.iloc[test_indices]

train, test = split_data(data,0.2)

x_train = train[['Age','Fever','Body Pain','Runny Nose','Breathing Difficulty','Loss of Taste']].to_numpy()
x_test = test[['Age','Fever','Body Pain','Runny Nose','Breathing Difficulty','Loss of Taste']].to_numpy()

Y_train = train[['Infected']].to_numpy().reshape(400,)
Y_test = test[['Infected']].to_numpy().reshape(100,)

clf = LogisticRegression()
clf.fit(x_train,Y_train)

#######################################################################################################################
Page = st.sidebar.selectbox('Do Select Your Page',('Infected Probability Rate','Covid Detection Via X-rays'))
if Page == 'Infected Probability Rate':
    st.title('Infected Probability Rate :syringe:')
    with st.container():
        st.write('---')
        st.write('##')
        left_column, right_column = st.columns(2)
        with left_column:
            Name = st.text_input('Enter Patient Name :')
            Age = st.number_input('Enter Your Age :',min_value=5, max_value=81, step=1)
            Gender = st.selectbox('Select Your Gender ',('Male','Female'))
            Fever = st.selectbox('Do you have Fever ?',('Yes','No'))

        with right_column:
            Body_Pain = st.selectbox('Do you have Body Pain ?',('Yes','No'))
            Runny_Nose = st.selectbox('Do you have Runny Nose ?',('Yes','No'))
            BDiff = st.selectbox('Do you have Breathing Difficulty ?',('Yes','No'))
            Loss = st.selectbox('Do you have Loss of Taste ?',('Yes','No'))

    Fever = [1 if Fever == 'Yes' else 0]
    Body_Pain = [1 if Body_Pain == 'Yes' else 0]
    Runny_Nose = [1 if Runny_Nose == 'Yes' else 0]
    BDiff = [1 if BDiff == 'Yes' else 0]
    Loss = [1 if Loss == 'Yes' else 0]

    with st.container():
        st.write('---')
        if st.button('Predict'):
            answer = clf.predict([[Age,Fever[0],Body_Pain[0],Runny_Nose[0],BDiff[0],Loss[0]]])[0]
            my_bar = st.progress(0)
            for pct in range(100):
                time.sleep(0.05)
                my_bar.progress(pct + 1)

            if answer == 0:
                st.subheader('{} has {} % No Infected Probability'.format(Name,round(clf.predict_proba([[Age,Fever[0],Body_Pain[0],Runny_Nose[0],BDiff[0],Loss[0]]])[0][1]*100,2)))
            else:
                st.subheader('{} has {} % Infected Probability'.format(Name,round(clf.predict_proba([[Age,Fever[0],Body_Pain[0],Runny_Nose[0],BDiff[0],Loss[0]]])[0][1]*100,2)))
else:
    st.title('Covid Detection via X-Rays :syringe:')
    with st.container():
        st.write('---')
        st.write('##')
        file_image = st.file_uploader("Upload X-Ray Image", type=["png","jpg","jpeg"])
        st.image(file_image)

        if file_image is not None:
            test_image = image.load_img(file_image,target_size=(64,64))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image,axis=0)
            result = Classifier.predict(test_image)
            print(result)

    with st.container():
        st.write('---')
        if st.button('Predict'):
            if result[0][0] == 1:
                st.subheader('It is COVID Negative X-ray')
            else:
                st.subheader('It is COVID Positive X-ray')
