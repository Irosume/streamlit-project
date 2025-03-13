import streamlit as st
import pandas as pd
import numpy as np
from random_forest import RandomForest

X_train = pd.read_csv("X_train.csv")

y_train = pd.read_csv("y_train.csv").squeeze()

if y_train.name in y_train:
    y_train = y_train[y_train != y_train.name]

@st.cache_resource
def train_model(X, y):
    
    clf = RandomForest()
    clf.fit(X, y)
    
    return clf

clf = train_model(X_train, y_train)

if "confirm" not in st.session_state:
    st.session_state.confirm = False
if "submitted" not in st.session_state:
    st.session_state.submitted = False

st.title("Prediksi Kelulusan dari Universitas dengan Model Random Forest ✨")
st.text(
    "Khawatir apakah anda akan lulus atau tidak dengan performa anda akhir-akhir ini?\nKami siap membantu anda dengan model prediksi kami!"
)
st.image(
    "https://images.pexels.com/photos/267885/pexels-photo-267885.jpeg?cs=srgb&dl=pexels-pixabay-267885.jpg&fm=jpg"
)

with st.sidebar.form("predict_form", clear_on_submit=True):
    st.title("Coba di sini! ✍️")
    st.session_state.umur = st.number_input(
        "Berapa umur anda?", min_value=1, max_value=100
    )
    st.session_state.ipk1 = st.number_input(
        "Berapakah IPK semester 1 anda?", min_value=0.0, max_value=4.0
    )
    st.session_state.ipk2 = st.number_input(
        "Berapakah IPK semester 2 anda?", min_value=0.0, max_value=4.0
    )
    st.session_state.absen = st.number_input(
        "Berapa kali anda masuk kuliah?", min_value=0, max_value=1000
    )
    st.session_state.belajar = st.number_input(
        "Berapa jam anda belajar per harinya?", min_value=0, max_value=24
    )
    st.session_state.tugas = st.number_input(
        "Berapa banyak tugas yang anda selesaikan?", min_value=0, max_value=100
    )
    st.session_state.motivasi = st.slider(
        "Dalam jangkauan 0 hingga 10, berapa tingkat motivasi anda?",
        min_value=0,
        max_value=10,
    )
    st.session_state.seminar = st.number_input(
        "Berapa banyak seminar yang telah anda ikuti?", min_value=0, max_value=10
    )
    st.session_state.jarak_rumah = st.number_input(
        "Berapa jarak tempat tinggal anda dari kampus? (Km)", min_value=0, max_value=100
    )
    st.session_state.nilai_ujian2 = st.number_input(
        "Berapa nilai ujian kedua dari terakhir anda sebelumnya?",
        min_value=0,
        max_value=100,
    )
    st.session_state.nilai_ujian1 = st.number_input(
        "Berapa nilai ujian terakhir anda sebelumnya?", min_value=0, max_value=100
    )
    submitted = st.form_submit_button("Submit")


if submitted:
    st.session_state.submitted = True
    st.session_state.confirm = False
    st.error("Apakah kamu yakin data yang kamu masukan sudah benar?")


if st.session_state.submitted:
    if st.button("Ya, data sudah benar."):
        st.session_state.confirm = True


if st.session_state.confirm:
    data = [
        st.session_state.umur,
        st.session_state.ipk1,
        st.session_state.ipk2,
        st.session_state.absen,
        st.session_state.belajar,
        st.session_state.tugas,
        st.session_state.motivasi,
        st.session_state.seminar,
        st.session_state.jarak_rumah,
        st.session_state.nilai_ujian2,
        st.session_state.nilai_ujian1,
    ]

    dataframe = pd.DataFrame([data], columns=[
        "Age", "GPA_Sem1", "GPA_Sem2", "Attendance", "Study_Hours", "Task_Completed",
        "Motivation_Level", "Seminars_Attended", "Distance_From_Campus", "Exam1_Score", "Exam2_Score"
    ])

    pred = clf.predict(dataframe)

    if pred == 1:
        st.success("Selamat! Anda diprediksi lulus.")
    else:
        st.error("Maaf. Anda diprediksi tidak lulus.")

    st.subheader("Hasil Prediksi:")
    st.write(f"Umur: {st.session_state.umur}")
    st.write(f"IPK Semester 1: {st.session_state.ipk1}")
    st.write(f"IPK Semester 2: {st.session_state.ipk2}")
    st.write(f"Kehadiran: {st.session_state.absen}")
    st.write(f"Waktu Belajar per Hari: {st.session_state.belajar}")
    st.write(f"Tugas Diselesaikan: {st.session_state.tugas}")
    st.write(f"Tingkat Motivasi: {st.session_state.motivasi}/10")
    st.write(f"Seminar Diikuti: {st.session_state.seminar}")
    st.write(f"Jarak Tempat Tinggal: {st.session_state.jarak_rumah} Km")
    st.write(f"Nilai Ujian Kedua dari Terakhir: {st.session_state.nilai_ujian2}")
    st.write(f"Nilai Ujian Terakhir: {st.session_state.nilai_ujian1}")
