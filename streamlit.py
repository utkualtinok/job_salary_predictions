import streamlit as st
import pandas as pd
import joblib
import time

# Streamlit Page Config
st.set_page_config(
    page_title="Maaş Tahmini",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Dataset
@st.cache_data
def get_data():
    return pd.read_csv(r"C:\Users\utku\Desktop\final_projects\Datasets\ds_salaries.csv")

# Load Model
@st.cache_resource
def load_model():
    return joblib.load(r"salary_prediction_model.joblib")

# CSS for Styling
st.markdown("""
    <style>
    body {
        background-color: #f7f9fc;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
    }
    .sub-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .highlight-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Page Title
st.markdown("<div class='main-title'>🚀 Veri Dünyasında Maaş Tahmini 🤑</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Keşfet, Tahmin Et, ve Daha Fazlasını Öğren!</div>", unsafe_allow_html=True)

# Side-by-Side Layout
tab1, tab2 = st.tabs(["Ana Sayfa", "Model Tahmini"])

# **Ana Sayfa**
with tab1:
    st.subheader("📊 Ana Sayfa")
    st.markdown(
        "Veri bilimci; bilgi odaklı keşifler yapabilen, bulunduğu ortama dair mevcut durumu tanımlayan, "
        "sınıflandırıcı olan ve durum hakkında gelecekle ilgili tahminleri yazılım kullanarak yapabilen kişidir. "
        "Peki bu sihirli meslek ne kadar kazanmaktadır? Gelin hep beraber inceleyelim."
    )

    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the column width ratios for centering
    with col2:
        st.image(
            "dspicture.webp",
            caption="Data Science Salary Predictions",
            use_column_width=True,  # Automatically adjust width to fit the column
        )

# **Model Tahmini**
with tab2:
    st.subheader("🤖 Model Sayfası")
    st.markdown("Aşağıdaki formu doldurarak maaş tahmininizi öğrenebilirsiniz.")

    # Inputs for Prediction
    col1, col2 = st.columns(2)
    with col1:
        experience_level = st.selectbox("Deneyim Seviyesi", ["Entry level", "Mid/Intermediate level", "Senior", "Executive level"])
        employment_type = st.selectbox("İş Türü", ["Freelancer", "Contractor", "Full-time", "Part-time"])
        company_size = st.selectbox("Şirket Büyüklüğü", ["Small", "Medium", "Large"])
    with col2:
        remote_ratio = st.selectbox("Uzaktan Çalışma Oranı", ["0", "50", "100"])
        work_year = st.selectbox("Yıl Seçiniz", ["2020", "2021", "2022", "2023"])
        same_country = st.selectbox("İş Yeri ve Çalışan Lokasyonu", ["Aynı Lokasyonda", "Farklı Lokasyonda"])

    # Load the Model
    model = load_model()

    # Prepare Input Data
    required_features = model.feature_names_
    input_data = {col: 0 for col in required_features}

    # Update Input Data Based on User Input
    input_data.update({
        'experience_level_EX': int(experience_level == "Executive level"),
        'experience_level_MI': int(experience_level == "Mid/Intermediate level"),
        'experience_level_SE': int(experience_level == "Senior"),
        'employment_type_FL': int(employment_type == "Freelancer"),
        'employment_type_FT': int(employment_type == "Full-time"),
        'employment_type_PT': int(employment_type == "Part-time"),
        'company_size_M': int(company_size == "Medium"),
        'company_size_S': int(company_size == "Small"),
        'work_year_2021': int(work_year == "2021"),
        'work_year_2022': int(work_year == "2022"),
        'work_year_2023': int(work_year == "2023"),
        'remote_ratio_50': int(remote_ratio == "50"),
        'remote_ratio_100': int(remote_ratio == "100"),
        'NEW_same_country_1': int(same_country == "Aynı Lokasyonda"),
    })

    input_df = pd.DataFrame([input_data])

    # Predict Salary
    if st.button("Maaşı Tahmin Et"):
        try:
            with st.spinner("Maaş Tahmini Yapılıyor..."):
                time.sleep(2)  # Simulate processing delay
                predicted_salary = model.predict(input_df)[0]

            # Display Prediction
            st.success(f"✅ Tahmin Edilen Maaş: ${round(predicted_salary, 2)}")

            # Animated "Pulse Effect" for the Salary
            animation_html = f"""
            <div style="text-align: center;">
                <div style="
                    font-size: 48px; 
                    font-weight: bold; 
                    color: #4CAF50; 
                    animation: pulse 2s infinite;
                ">
                    Maaş: ${round(predicted_salary, 2)}
                </div>
                <style>
                    @keyframes pulse {{
                        0% {{ transform: scale(1); opacity: 1; }}
                        50% {{ transform: scale(1.1); opacity: 0.8; }}
                        100% {{ transform: scale(1); opacity: 1; }}
                    }}
                </style>
            </div>
            """
            st.components.v1.html(animation_html, height=200)

        except Exception as e:
            st.error("Tahmin sırasında bir hata oluştu.")
            st.write(f"Error details: {e}")
