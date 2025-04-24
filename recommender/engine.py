import pickle
import pandas as pd
import shap
import numpy as np

from .utils import ease_map, generate_shap_explainer_and_scaler

def generate_recommendations(model, label_encoders: dict, ad_sample: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    
    scaler, explainer = generate_shap_explainer_and_scaler(model, df)
    
    ad_sample_scaled = scaler.transform(ad_sample.drop(columns=['roi']))

    other_targets = ['impressions', 'clicks', 'ctr', 'conversions', 'conversion_rate']
    df = df.drop(columns=other_targets)

    X = df.drop(columns=['roi'])

    base_roi = model.predict(ad_sample_scaled)[0]

    shap_values = explainer(ad_sample_scaled)

    shap_df = pd.DataFrame({
        "feature": X.columns,
        "shap_value": shap_values.values[0],
    })

    worst_features = shap_df.sort_values(by="shap_value").head(10)

    suggestions = []

    for _, row in worst_features.iterrows():
        feature = row['feature']

        # Caso seja uma feature categórica (label encoded)
        if feature.replace('_encoded', '') in label_encoders:
            encoder_label = feature.replace('_encoded', '')
            current_feature_value = ad_sample[feature].values[0].astype(int)
            current_label = label_encoders[encoder_label].inverse_transform([current_feature_value])[0]
            best_change = None

            for label in label_encoders[encoder_label].classes_:
                if label == current_feature_value: continue

                new_sample = ad_sample.copy()
                new_sample[feature] = label_encoders[encoder_label].transform([label])[0]

                new_roi = model.predict(scaler.transform(new_sample.drop(columns=['roi'])))[0]
                delta_roi = new_roi - base_roi
                
                if delta_roi > 0 and (best_change is None or delta_roi > best_change["delta_roi"]):
                    best_change = {
                        "feature": feature,
                        "from": current_label,
                        "to": label,
                        "roi": new_roi,
                        "delta_roi": delta_roi,
                        "ease_score": ease_map.get(feature, ease_map["default"]),
                    }

            if best_change:
                suggestions.append(best_change)
        
        else:
            current_feature_value = ad_sample[feature].values[0]
            best_change = None

            # Gera 15 perturbações entre -30% e +30%
            for pct in np.linspace(-0.3, 0.3, 15):
                new_value = current_feature_value * (1 + pct)

                if new_value < 0:
                    continue

                new_sample = ad_sample.copy()
                new_sample[feature] = new_value
                new_roi = model.predict(scaler.transform(new_sample.drop(columns=['roi'])))[0]
                delta_roi = new_roi - base_roi

                if delta_roi > 0 and (best_change is None or delta_roi > best_change["delta_roi"]):
                    best_change = {
                        "feature": feature,
                        "from": round(current_feature_value, 2),
                        "to": round(new_value, 2),
                        "roi": new_roi,
                        "delta_roi": delta_roi,
                        "ease_score": ease_map.get(feature, ease_map["default"]),
                        "pct_change": f"{pct*100:.1f}%"
                    }

            if best_change:
                suggestions.append(best_change)
    
    return suggestions, base_roi