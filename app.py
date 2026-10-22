
import os
import joblib
import pandas as pd
import streamlit as st

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Haven | House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = "models/house_price_model.pkl"

# ---------------- HOUSE IMAGES ----------------
HERO_IMAGE = (
    "https://images.unsplash.com/photo-1600596542815-ffad4c1539a9"
    "?auto=format&fit=crop&w=1800&q=85"
)

HOUSE_IMAGES = [
    (
        "Modern luxury",
        "https://images.unsplash.com/photo-1600607687939-ce8a6c25118c"
        "?auto=format&fit=crop&w=800&q=80"
    ),
    (
        "Contemporary living",
        "https://images.unsplash.com/photo-1600210492486-724fe5c67fb0"
        "?auto=format&fit=crop&w=800&q=80"
    ),
    (
        "Elegant exterior",
        "https://images.unsplash.com/photo-1600585154340-be6161a56a0c"
        "?auto=format&fit=crop&w=800&q=80"
    ),
]

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

:root {
    --navy: #152c2b;
    --green: #214b40;
    --gold: #d5ad70;
    --cream: #f7f5ef;
    --muted: #68766f;
}

.stApp {
    background: var(--cream);
    color: var(--navy);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1240px;
    padding-top: 1.4rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    font-size: 1.45rem;
    font-weight: 700;
    letter-spacing: -0.8px;
    color: var(--navy);
    padding: 0.3rem 0 1.1rem;
}

.brand span {
    color: #b38a4f;
}

.hero {
    position: relative;
    overflow: hidden;
    min-height: 350px;
    padding: 3.2rem 3rem;
    border-radius: 22px;
    background:
        linear-gradient(90deg,
        rgba(13, 31, 29, .94) 0%,
        rgba(13, 31, 29, .77) 48%,
        rgba(13, 31, 29, .08) 100%),
        url("https://images.unsplash.com/photo-1600596542815-ffad4c1539a9?auto=format&fit=crop&w=1800&q=85")
        center 55% / cover;
    display: flex;
    flex-direction: column;
    justify-content: center;
    margin-bottom: 1.6rem;
}

.eyebrow {
    color: #e0c28f;
    font-size: .76rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}

.hero h1 {
    font-family: 'Playfair Display', Georgia, serif;
    color: #fff;
    font-size: clamp(2.3rem, 5vw, 4rem);
    line-height: 1.12;
    max-width: 590px;
    margin: 1rem 0;
}

.hero p {
    color: #e0e8e2;
    font-size: 1rem;
    line-height: 1.8;
    max-width: 460px;
}

.hero-tag {
    color: #f2dfba;
    font-size: .8rem;
    margin-top: 1rem;
}

.section-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--navy);
    margin-top: .5rem;
}

.section-subtitle {
    color: var(--muted);
    font-size: .92rem;
    margin: .25rem 0 1.3rem;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #fff;
    border: 1px solid #e8e7df;
    border-radius: 16px;
}

div[data-testid="stWidgetLabel"] p {
    color: #344b43;
    font-weight: 600;
    font-size: .87rem;
}

div[data-baseweb="select"] > div,
div[data-testid="stNumberInput"] input {
    background: #fbfaf7;
    border-radius: 9px;
    border-color: #e4e5dc;
}

.stButton > button {
    background: var(--green);
    color: white;
    border: 1px solid var(--green);
    border-radius: 10px;
    padding: .8rem 1.3rem;
    font-weight: 700;
    min-height: 3rem;
    transition: .2s ease;
}

.stButton > button:hover {
    background: #17372f;
    color: #fff;
    border-color: var(--gold);
    transform: translateY(-1px);
}

.result-card {
    background: #183b32;
    color: white;
    border-radius: 18px;
    padding: 1.7rem 2rem;
    margin-top: 1rem;
    border: 1px solid #34584c;
}

.result-label {
    color: #d9c49c;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-size: .75rem;
    font-weight: 700;
}

.result-price {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: clamp(2rem, 5vw, 3.1rem);
    color: white;
    font-weight: 600;
    margin: .4rem 0;
}

.result-note {
    color: #d7e3db;
    font-size: .85rem;
    line-height: 1.6;
}

.info-chip {
    background: white;
    border: 1px solid #e8e7df;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: .8rem;
}

.info-chip strong {
    display: block;
    color: var(--green);
    font-size: 1.3rem;
}

.info-chip span {
    color: var(--muted);
    font-size: .76rem;
}

.gallery-caption {
    color: var(--navy);
    font-weight: 700;
    font-size: .9rem;
    padding-top: .4rem;
}

.footer {
    border-top: 1px solid #e2e0d7;
    margin-top: 3rem;
    padding-top: 1.2rem;
    text-align: center;
    color: #718078;
    font-size: .8rem;
}

@media (max-width: 700px) {
    .hero {
        padding: 2rem 1.3rem;
        min-height: 330px;
    }
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- BRAND ----------------
st.markdown(
    '<div class="brand">HAVEN<span>.</span> '
    '<span style="color:#68766f;font-size:.72rem;'
    'font-weight:500;letter-spacing:1.5px;">PROPERTY INTELLIGENCE</span></div>',
    unsafe_allow_html=True
)

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">A smarter way to find your home</div>
    <h1>Every home has<br>a value. Discover yours.</h1>
    <p>
        Explore an estimated property value using machine learning
        and the details that make your home unique.
    </p>
    <div class="hero-tag">PROPERTY ESTIMATOR &nbsp; • &nbsp; AI-POWERED</div>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Run `python train.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ---------------- INPUT SECTION ----------------
st.markdown(
    '<div class="section-title">Tell us about the property</div>'
    '<div class="section-subtitle">Adjust the details below to estimate '
    'a property price.</div>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.markdown("#### 01 &nbsp; Property dimensions")
    st.caption("Size, layout and practical features")

    c1, c2, c3 = st.columns(3)

    with c1:
        area = st.number_input(
            "Built-up area (sq. ft.)",
            min_value=100, max_value=20000,
            value=4000, step=100
        )

    with c2:
        bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6], index=2)

    with c3:
        bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)

    c4, c5 = st.columns(2)

    with c4:
        stories = st.selectbox("Number of stories", [1, 2, 3, 4], index=1)

    with c5:
        parking = st.selectbox("Parking spaces", [0, 1, 2, 3], index=1)

    st.markdown("---")
    st.markdown("#### 02 &nbsp; Property amenities")
    st.caption("Choose the facilities available at the property")

    c1, c2, c3 = st.columns(3)

    with c1:
        mainroad = st.selectbox("Main road access", ["yes", "no"])
        guestroom = st.selectbox("Guestroom", ["yes", "no"])
        basement = st.selectbox("Basement", ["yes", "no"])

    with c2:
        airconditioning = st.selectbox("Air conditioning", ["yes", "no"])
        hotwaterheating = st.selectbox("Hot water heating", ["yes", "no"])
        prefarea = st.selectbox("Preferred residential area", ["yes", "no"])

    with c3:
        furnishingstatus = st.selectbox(
            "Furnishing status",
            ["furnished", "semi-furnished", "unfurnished"]
        )

        st.markdown(
            '<div style="background:#f5f3ec;padding:1rem;'
            'border-radius:10px;margin-top:.6rem;color:#5c6e64;'
            'font-size:.82rem;line-height:1.6;">'
            '<b>Tip</b><br>Use the actual property details where possible '
            'for a more meaningful estimate.</div>',
            unsafe_allow_html=True
        )

    st.markdown("")
    predict = st.button(
        "Estimate property value  →",
        type="primary",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------
if predict:
    house = pd.DataFrame([{
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "parking": parking,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }])

    prediction = float(model.predict(house)[0])

    st.markdown("### Your property estimate")

    if prediction <= 0:
        st.warning(
            "The model returned a non-positive estimate. "
            "Try different property details."
        )
    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated property value</div>
            <div class="result-price">₹ {prediction / 100000:,.2f} lakh</div>
            <div class="result-note">
                Estimated from the selected property features.
                This is a machine-learning estimate, not a formal valuation
                or a guaranteed selling price.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f'<div class="info-chip"><strong>{area:,}</strong>'
                '<span>Square feet</span></div>',
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f'<div class="info-chip"><strong>{bedrooms}</strong>'
                '<span>Bedrooms</span></div>',
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                f'<div class="info-chip"><strong>{parking}</strong>'
                '<span>Parking spaces</span></div>',
                unsafe_allow_html=True
            )

        # Download the details and prediction
        report = house.copy()
        report["estimated_price"] = round(prediction, 2)

        st.download_button(
            "Download property estimate (CSV)",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name="house_price_estimate.csv",
            mime="text/csv"
        )

# ---------------- INSPIRATION GALLERY ----------------
st.markdown("")
st.markdown(
    '<div class="section-title">Find your kind of home</div>'
    '<div class="section-subtitle">A little inspiration for your next address.</div>',
    unsafe_allow_html=True
)

gallery = st.columns(3)

for column, (caption, image_url) in zip(gallery, HOUSE_IMAGES):
    with column:
        st.image(image_url, use_container_width=True)
        st.markdown(
            f'<div class="gallery-caption">{caption}</div>',
            unsafe_allow_html=True
        )

st.markdown("""
<div class="footer">
    HAVEN. &nbsp; | &nbsp; House Price Prediction &nbsp; | &nbsp;
    Built with Python, Scikit-learn & Streamlit
    <br><br>
    Estimates depend on the dataset and model quality. Actual market
    prices can differ significantly.
</div>
""", unsafe_allow_html=True)

#streamlit run app.py