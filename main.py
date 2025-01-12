import streamlit as st
import pandas as pd

from model import predict

st.title("نموذج إدخال بيانات السكري")

st.header("يرجى إدخال القيم التالية:")

Pregnancies = st.slider("عدد الحمل", min_value=0, max_value=20, step=1, value=1)
Glucose = st.slider("مستوى الجلوكوز", min_value=0, max_value=200, step=1, value=80)
BloodPressure = st.slider("ضغط الدم", min_value=0, max_value=200, step=1, value=70)
SkinThickness = st.slider("سمك الجلد", min_value=0, max_value=100, step=1, value=20)
Insulin = st.slider("الأنسولين", min_value=0, max_value=500, step=1, value=80)
BMI = st.slider("مؤشر كتلة الجسم (BMI)", min_value=10.0, max_value=50.0, step=0.1, value=25.0)
DiabetesPedigreeFunction = st.slider("وظيفة علم الوراثة للسكري", min_value=0.0, max_value=2.5, step=0.01, value=0.5)
Age = st.slider("العمر", min_value=18, max_value=120, step=1, value=30)

data = {
    'Pregnancies': [Pregnancies],
    'Glucose': [Glucose],
    'BloodPressure': [BloodPressure],
    'SkinThickness': [SkinThickness],
    'Insulin': [Insulin],
    "BMI": [BMI],
    "DiabetesPedigreeFunction": [DiabetesPedigreeFunction],
    "Age": [Age]
}

st.subheader("البيانات المدخلة:")
st.write(pd.DataFrame(data))

if st.button("إرسال البيانات"):
    st.success("تم إرسال البيانات بنجاح!")
    result = predict(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    st.write(f"result :{result}")
    if result == "negative":
        st.balloons()
