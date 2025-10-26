import joblib
import pandas as pd

_model = None
_feats = None

def load_artifacts(model_path='models/model.pkl', feats_path='models/feats.pkl'):
    global _model, _feats
    if _model is None:
        _model = joblib.load(model_path)
    if feats_path:
        _feats = joblib.load(feats_path)

def predict_ai_score(zip_features_df: pd.DataFrame) -> pd.DataFrame:
    # zip_features_df should match the training schema (or pass through _feats)
    X = _feats.transform(zip_features_df) if _feats else zip_features_df
    # choose .predict_proba[:,1] or .predict(...) depending on model
    score = getattr(_model, "predict_proba", None)
    if score:
        ai = _model.predict_proba(X)[:, 1]
    else:
        ai = _model.predict(X)
    out = zip_features_df[['zip']].copy()
    out['ai_score'] = ai
    return out
