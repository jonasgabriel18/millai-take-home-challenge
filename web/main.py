import streamlit as st
import requests

platform_possible = ["Meta", "LinkedIn", "TikTok", "YouTube"]
ad_type_possible = ["Video", "Image", "Carousel", "Story"]
industry_possible = ["Retail", "Tech", "Finance", "B2B Technology", "Entertainment", "E-commerce", "Beauty", "Gaming",
                      "Professional Services", "Cosmetics", "Electronics", "Health", "SaaS", "Education", "Home Goods",
                      "Travel", "Fitness", "Food & Beverage", "Automotive", "Fashion", "Jewelry"]

campaign_objective_possible = ["Conversion", "Awareness", "Consideration", "Lead Generation", "App Install"]

cta_type_possible = [
    'Shop Now', 'Learn More', 'Request Demo', 'Shop Collection', 'Watch Now',
    'Download Report', 'Get Demo', 'Discover More', 'See Collection', 'Watch Webinar',
    'Get Guide', 'Browse Collection', 'Subscribe Now', 'Sign Up', 'Subscribe',
    'Download Now', 'Contact Us', 'Book Now', 'Register Now', 'Join Now',
    'Order Now', 'Book Test Drive', 'Download Guide'
]


st.set_page_config(page_title="Ad Optimizer", layout="wide")
st.title("🎯 Ad Optimization Assistant")

API_URL = "http://localhost:8000"

# Função para obter o sample, se necessário
if "random_sample" not in st.session_state:
    st.session_state.random_sample = {}

# Botão para gerar novo sample
if st.button("🎲 Generate Random Sample"):
    with st.spinner("Fetching sample..."):
        res = requests.get(f"{API_URL}/generate-random-sample")
        if res.status_code == 200:
            st.session_state.random_sample = res.json()
            st.rerun()
        else:
            st.error(f"Error: {res.text}")

sample = st.session_state.random_sample

with st.form("ad_form"):
    st.subheader("📋 Advertisement Parameters")

    col1, col2, col3 = st.columns(3)
    with col1:
        duration_seconds = st.number_input("Duration (seconds)", value=sample.get("duration_seconds", 12.0))
        text_to_image_ratio = st.number_input("Text to Image Ratio", value=sample.get("text_to_image_ratio", 0.23))
        face_count = st.number_input("Face Count", value=sample.get("face_count", 2))
        word_count = st.number_input("Word Count", value=sample.get("word_count", 32))
        music_tempo = st.number_input("Music Tempo", value=sample.get("music_tempo", 112.0))
        speech_pace = st.number_input("Speech Pace", value=sample.get("speech_pace", 120.0))
        roi = 0.0  # oculto ou fixo

    with col2:
        brightness_score = st.slider("Brightness Score", 0, 100, value=int(sample.get("brightness_score", 70)))
        logo_size_ratio = st.number_input("Logo Size Ratio", value=sample.get("logo_size_ratio", 0.11))
        sentiment_score = st.slider("Sentiment Score", 0.0, 1.0, value=float(sample.get("sentiment_score", 0.75)), step=0.01)
        has_human_face = st.selectbox("Has Human Face?", [0, 1], index=[0, 1].index(sample.get("has_human_face", 0)))
        color_palette_primary_R = st.slider("Color R", 0, 255, value=int(sample.get("color_palette_primary_R", 200)))
        color_palette_primary_G = st.slider("Color G", 0, 255, value=int(sample.get("color_palette_primary_G", 220)))

    with col3:
        color_palette_primary_B = st.slider("Color B", 0, 255, value=int(sample.get("color_palette_primary_B", 150)))
        platform = st.selectbox("Platform", platform_possible, index=platform_possible.index(sample.get("platform", "Meta")))

        ad_type = st.selectbox("Ad Type", ad_type_possible, index=ad_type_possible.index(sample.get("ad_type", "Video")))

        industry = st.selectbox("Industry", industry_possible, index=industry_possible.index(sample.get("industry", "Retail")))

        campaign_objective = st.selectbox("Campaign Objective", campaign_objective_possible, index=campaign_objective_possible.index(sample.get("campaign_objective", "Conversion")))
        target_audience_gender = st.selectbox("Audience Gender", ["female", "all"], index=["female", "all"].index(sample.get("target_audience_gender", "all")))
        aspect_ratio = st.selectbox("Aspect Ratio", ["1:1", "16:9", "9:16", "4:5", "4:3"], index=["1:1", "16:9", "9:16", "4:5", "4:3"].index(sample.get("aspect_ratio", "1:1")))
        cta_type = st.selectbox("CTA Type", cta_type_possible, index=cta_type_possible.index(sample.get("cta_type", "Learn More")))

    submitted = st.form_submit_button("📈 Generate Recommendations")
    if submitted:
        payload = {
            "duration_seconds": duration_seconds,
            "brightness_score": brightness_score,
            "text_to_image_ratio": text_to_image_ratio,
            "logo_size_ratio": logo_size_ratio,
            "has_human_face": has_human_face,
            "face_count": face_count,
            "sentiment_score": sentiment_score,
            "word_count": word_count,
            "music_tempo": music_tempo,
            "speech_pace": speech_pace,
            "roi": roi,
            "color_palette_primary_R": color_palette_primary_R,
            "color_palette_primary_G": color_palette_primary_G,
            "color_palette_primary_B": color_palette_primary_B,
            "platform": platform,
            "ad_type": ad_type,
            "industry": industry,
            "campaign_objective": campaign_objective,
            "target_audience_gender": target_audience_gender,
            "aspect_ratio": aspect_ratio,
            "cta_type": cta_type,
        }

        with st.spinner("Generating..."):
            res = requests.post(f"{API_URL}/generate-recommendations", json=payload)
            if res.status_code == 200:
                st.success("✅ Recommendations generated successfully!")
                st.markdown(f"### Original ROI: **{round(res.json()['original_roi'], 2)}**")
                for r in res.json()['suggestions']:
                    st.markdown(f"📌 **{r['feature']}**: Change from **{r['from']}** to **{r['to']}** → Expected ROI: **{round(r['roi'], 2)}** (+{round(r['delta_roi'], 2)})")
            else:
                st.error(f"❌ Error: {res.text}")