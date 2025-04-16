from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('ctr').getOrCreate()
df = spark.read.csv('/content/drive/MyDrive/synthetic_ctr_data.csv',header = True,inferSchema = True)
df.repartition(8)
df.write.parquet("/content/sample_data/synthetic_ctr_data.parquet")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, RobustScaler, TargetEncoder
from sklearn.metrics import (classification_report, confusion_matrix,
                            roc_auc_score)
from imblearn.pipeline import Pipeline  as impipeline
from sklearn.pipeline import Pipeline as skpipeline
from imblearn.combine import SMOTETomek
import lightgbm as lgb
import optuna
from optuna.samplers import NSGAIISampler

# Configuration
RANDOM_STATE = 42
DATA_PATH = '/content/sample_data/synthetic_ctr_data.parquet'

# ========== Data Loading & Feature Engineering ==========
def load_and_preprocess(path):
    df = pd.read_parquet(path)

    # Clean data
    df = (df.drop_duplicates()
          .dropna()
          .drop(columns=['id'])  # Remove non-predictive ID
          )

    # Temporal features
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df = df.drop(columns=['timestamp'])

    # Interaction features
    if {'page_views', 'time_spent'}.issubset(df.columns):
        df['view_time_ratio'] = df['time_spent'] / (df['page_views'] + 1e-6)

    return df

df = load_and_preprocess(DATA_PATH)

X = df.drop(columns=['click'])
y = df['click']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = X_train.select_dtypes(include=np.number).columns.tolist()

prep_1 = ColumnTransformer([
    ('cat', TargetEncoder(smooth=50), cat_cols),
    ('num', skpipeline([('quantile', QuantileTransformer(n_quantiles=1000))]), num_cols)
], remainder='passthrough')

# Resampling pipeline
processing_pipeline = impipeline([
    ('preprocessing' , prep_1),
    ('smotetomek', SMOTETomek(sampling_strategy=0.5, random_state=RANDOM_STATE))
])

X_train_proc, y_train_proc = processing_pipeline.fit_resample(X_train, y_train)

class Objective:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __call__(self, trial):
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': trial.suggest_categorical('boosting_type', ['dart', 'goss']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-2, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 18, 21),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'n_estimators': trial.suggest_int('n_estimators', 300, 600),
            'class_weight': 'balanced',
            'random_state': RANDOM_STATE
        }

        model = lgb.LGBMClassifier(**params)
        scores_f1 = cross_val_score(model, self.X, self.y, cv=2,
                               scoring='f1', n_jobs=-1)
        return scores_f1.mean()

study = optuna.create_study(direction='maximize')
study.optimize(Objective(X_train_proc, y_train_proc), n_trials=5)
best_params = study.best_params
# index  = 0
# params = None
# value = 0
# for index, score_list in enumerate(study.best_trials):
#     val= score_list.values[1]
#     if value < val:
#         params = score_list.params


best_model = lgb.LGBMClassifier(**best_params)
best_model.fit(X_train_proc, y_train_proc)

def evaluate_model(model, t1, X_test, y_test):
    X_test_proc = t1.transform(X_test)

    y_pred = model.predict(X_test_proc)
    y_proba = model.predict_proba(X_test_proc)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print(f"\nROC AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f' confusion matrix : {confusion_matrix(y_test, y_pred)}')

evaluate_model(best_model, prep_1, X_test, y_test)
''' 
for test 

''' 

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic CTR data
np.random.seed(42)
n_samples = 50_000

data = {
    'hour': np.random.randint(0, 24, n_samples),
    'user_id': [f'USER_{i:05d}' for i in np.random.randint(1, 15000, n_samples)],
    'user_age': np.clip(np.random.normal(35, 10, n_samples).astype(int), 18, 80),
    'user_gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04]),
    'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples, p=[0.6, 0.35, 0.05]),
    'device_os': np.random.choice(['Android', 'iOS', 'Windows', 'MacOS'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
    'device_browser': np.random.choice(['Chrome', 'Safari', 'Firefox', 'Edge'], n_samples, p=[0.6, 0.25, 0.1, 0.05]),
    'ad_campaign_id': [f'CAMP_{i:03d}' for i in np.random.randint(1, 50, n_samples)],
    'ad_creative_type': np.random.choice(['Banner', 'Video', 'Native', 'Popup'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'ad_position': np.random.choice(['Top', 'Middle', 'Sidebar', 'Bottom'], n_samples, p=[0.3, 0.2, 0.25, 0.25]),
    'site_id': [f'SITE_{i:04d}' for i in np.random.randint(1, 500, n_samples)],
    'site_domain': np.random.choice(['news', 'shopping', 'social', 'entertainment'], n_samples, p=[0.3, 0.4, 0.2, 0.1]),
    'site_category': np.random.choice(['technology', 'fashion', 'sports', 'politics'], n_samples, p=[0.4, 0.3, 0.2, 0.1]),
    'click': np.zeros(n_samples)  # To be populated
}

# Create realistic click patterns
df = pd.DataFrame(data)

# Generate click probabilities based on feature combinations
click_proba = (
    + 0.3 * (df['ad_position'] == 'Top')
    + 0.2 * (df['device_os'] == 'Android')
    - 0.1 * (df['user_age'] > 50)
    + 0.4 * (df['site_domain'] == 'shopping')
    + 0.25 * (df['ad_creative_type'] == 'Video')
    - 0.15 * (df['hour'].between(2, 5))
    + np.random.normal(0, 0.2, n_samples)
)

# Convert to binary clicks with 15% base CTR
df['click'] = (click_proba > np.percentile(click_proba, 85)).astype(int)

# Split data
X = df.drop('click', axis=1)
y = df['click']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print("Dataset shape:", df.shape)
print(f"CTR: {y.mean():.2%}")
print("\nSample data:")

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import QuantileTransformer, FunctionTransformer, TargetEncoder
from imblearn.pipeline import make_pipeline

# Identify feature types
cat_features = ['user_gender', 'device_type', 'device_os', 'device_browser',
               'ad_campaign_id', 'ad_creative_type', 'ad_position',
               'site_id', 'site_domain', 'site_category']

num_features = ['hour', 'user_age']

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', TargetEncoder(smooth=50), cat_features),
    ('num', make_pipeline(
        QuantileTransformer(n_quantiles=1000, output_distribution='normal'),
    ), num_features)
])

# Full pipeline with resampling
pipeline = make_pipeline(
    preprocessor,
    SMOTETomek(sampling_strategy=0.5, random_state=42),
    lgb.LGBMClassifier(**params)  # Use your best params from previous training
)

# Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("Evaluation Metrics:")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f'confusion matrix : {confusion_matrix(y_test, y_pred)}')
