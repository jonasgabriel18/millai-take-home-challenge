from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np
import joblib

from recommender.engine import generate_recommendations
from recommender.utils import generate_random_sample

app = FastAPI()

with open ("./models/encoders/label_encoders.pkl", "rb") as f:
    label_encoders = joblib.load(f)

with open("./models/trained_models/xgb_model.pkl", "rb") as f:
    model = pickle.load(f)

df = pd.read_csv("data/processed/sample_feature_engineered.csv")

class AdSample(BaseModel):
    duration_seconds: float
    brightness_score: float
    text_to_image_ratio: float
    logo_size_ratio: float
    has_human_face: int
    face_count: int
    sentiment_score: float
    word_count: int
    music_tempo: float
    speech_pace: float
    roi: float
    color_palette_primary_R: int
    color_palette_primary_G: int
    color_palette_primary_B: int
    platform: str
    ad_type: str
    industry: str
    campaign_objective: str
    target_audience_gender: str
    aspect_ratio: str
    cta_type: str

@app.post("/generate-recommendations")
def generate(ad: AdSample):
    try:
        ad_dict = ad.model_dump()
        ad_df = pd.DataFrame([ad_dict])
        # print(ad_df)
        categorial_columns = ad_df.select_dtypes(include=['object']).columns.tolist()
        # print(label_encoders)

        for column, encoder in label_encoders.items():
            ad_df[column] = encoder.transform(ad_df[column])
            ad_df[f"{column}_encoded"] = ad_df[column]
            ad_df.drop(columns=[column], inplace=True)


        ad_df['roi'] = 0

        suggestions, base_roi = generate_recommendations(model, label_encoders, ad_df, df)
        cleaned_suggestions = []

        for s in suggestions:
            cleaned_suggestions.append({
                k: (
                    float(v) if isinstance(v, (np.float32, np.float64))
                    else int(v) if isinstance(v, (np.int32, np.int64))
                    else str(v) if isinstance(v, np.generic)
                    else v
                )
                for k, v in s.items()
            })

        return {"suggestions": cleaned_suggestions, "original_roi": float(base_roi)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/generate-random-sample")
def generate_random_sample_api():
    try:
        sample = generate_random_sample(df, label_encoders)
        return sample.to_dict(orient="records")[0]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)