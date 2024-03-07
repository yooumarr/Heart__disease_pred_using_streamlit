import numpy as np
import pickle 
import streamlit as st

#loading our saved model
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def heart_predict(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float32)

    #reshaping numpy arrays since we are predicting for only on one instance

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)


    if prediction[0] == 0:
      return"Your heart is healthy"
    else:
      return"Your heart is not healthy" 
      
      
def main():
    st.title("Heart disease prediction")

    #getting inputs from a user    
    age = st.text_input("Age")
    sex = st.text_input("Sex")
    cp = st.text_input("Chest pain type")
    trestbps = st.text_input("Resting blood pressure")
    chol = st.text_input("Serum cholestoraL")
    fbs = st.text_input("Fasting blood sugar")
    restecg = st.text_input("Resting electrocardiographic results")
    thalach = st.text_input("Maximum heart rate achieved")
    exang = st.text_input("Exercise induced angina")
    oldpeak = st.text_input("Oldpeak (ST depression induced by exercise relative to rest)")
    slope = st.text_input("The slope of the peak exercise ST segment")
    ca = st.text_input("Number of major vessels (0-3) colored by flourosopy")
    thal = st.text_input("Thal")
    
    
    #code forpediction
    diagnosis = ''
    
    if st.button('Heart health test result'):
        diagnosis = heart_predict([age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
        main()
