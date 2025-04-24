import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def generate_shap_explainer_and_scaler(model, df: pd.DataFrame):
    """
    Generate Standard Scaler and SHAP explainer for the model using the provided DataFrame.
    """
    not_target_columns = ['impressions', 'clicks', 'ctr', 'conversions', 'conversion_rate', 'roi']
    X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=not_target_columns), df['roi'], test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    explainer = shap.Explainer(model, X_train_scaled_df)
    
    return scaler, explainer

def generate_random_sample(df: pd.DataFrame, label_encoders) -> pd.DataFrame:
    """
    Generate a random sample of the DataFrame
    """

    indices = np.random.choice(df.index, size=(1, df.shape[1]), replace=True)
    ad_sample = pd.DataFrame(data=df.to_numpy()[indices, np.arange(len(df.columns))], 
                        columns=df.columns)
    
    for col, encoder in label_encoders.items():
        encoded_col = col + "_encoded"
        if encoded_col in ad_sample:
            ad_sample[col] = encoder.inverse_transform([int(ad_sample[encoded_col].iloc[0])])[0]
            del ad_sample[encoded_col]
    
    return ad_sample

ease_map = {
    "duration_seconds": 1.5,
    "brightness_score": 1,
    "text_to_image_ratio": 1,
    "logo_size_ratio": 1,
    "has_human_face": 2,
    "face_count": 2,
    "sentiment_score": 1,
    "word_count": 1,
    "music_tempo": 1.5,
    "speech_pace": 1.5,
    "color_palette_primary_R": 1,
    "color_palette_primary_G": 1,
    "color_palette_primary_B": 1,
    "platform_encoded": 3,
    "industry_encoded": 3,
    "campaign_objective_encoded": 3,
    "target_audience_gender_encoded": 2,
    "aspect_ratio_encoded": 1,
    "cta_type_encoded": 1,
    "default": 1
}