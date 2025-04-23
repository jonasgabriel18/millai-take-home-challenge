import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

def feature_selection(df: pd.DataFrame):
    """
    Selects relevant features from the DataFrame

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with selected features.
        dict: Dictionary of label encoders for categorical features.
    """
    
    df_features = df[['platform', 'ad_type', 'industry', 'campaign_objective', 'target_audience_gender', 
                  'duration_seconds', 'aspect_ratio', 'brightness_score', 'color_palette_primary',
                  'text_to_image_ratio', 'logo_size_ratio', 'has_human_face', 'face_count',
                  'sentiment_score', 'word_count', 'cta_type', 'music_tempo', 'speech_pace',
                  'impressions', 'clicks', 'ctr', 'conversions', 'conversion_rate', 'roi']]
    
    rgb_values = df_features['color_palette_primary'].str.strip('[]').str.split(',', expand=True).astype(int)
    df_features['color_palette_primary_R'] = rgb_values[0]
    df_features['color_palette_primary_G'] = rgb_values[1]
    df_features['color_palette_primary_B'] = rgb_values[2]

    df_features['has_human_face'] = df_features['has_human_face'].astype(int)

    df_features.drop(columns=['color_palette_primary'], inplace=True)

    categorial_columns = df_features.select_dtypes(include=['object']).columns.tolist()

    label_encoders = {}
    for col in categorial_columns:
        le = LabelEncoder()
        df_features[col + '_encoded'] = le.fit_transform(df_features[col])
        label_encoders[col] = le

    df_features.drop(columns=categorial_columns, inplace=True)

    return df_features, label_encoders

if __name__ == "__main__":
    df = pd.read_csv("data/processed/sample_treated.csv")
    df_features, label_encoders = feature_selection(df)

    joblib.dump(label_encoders, 'models/encoders/label_encoders.pkl')
    df_features.to_csv('data/processed/sample_feature_engineered.csv', index=False)