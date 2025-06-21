# train_model.py

import pandas as pd
import pickle
import os

from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# 1. Load dataset
data = pd.read_csv('dataset_mahasiswa_do.csv')

# 2. Preprocessing
num_cols = data.select_dtypes(include=['int64', 'float64']).columns
cat_cols = data.select_dtypes(include=['object']).columns

data[num_cols] = SimpleImputer(strategy='mean').fit_transform(data[num_cols])
data[cat_cols] = SimpleImputer(strategy='most_frequent').fit_transform(data[cat_cols])

# 3. Fitur & target
fitur = ['ipk', 'kehadiran', 'penghasilan_orang_tua', 'motivasi_belajar', 'usia', 'semester', 'beban_sks']
X = pd.DataFrame(data, columns=fitur)
y = data['status_do']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

# 5. Feature selector
selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
selector_model.fit(X_train, y_train)
selector = SelectFromModel(selector_model, threshold='median')
X_train_sel = selector.transform(X_train)

# 6. GridSearch + SMOTE
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
])

param_grid = {
    'rf__n_estimators': [100],
    'rf__max_depth': [None],
    'rf__min_samples_split': [2],
    'rf__min_samples_leaf': [1]
}

grid = GridSearchCV(pipeline, param_grid, cv=3)
grid.fit(X_train_sel, y_train)

# 7. Buat pipeline akhir: imputasi + seleksi fitur + model
final_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('selector', selector),
    ('model', grid.best_estimator_)
])

# Fit pipeline ke data asli (X, y)
final_pipeline.fit(X, y)

# 8. Simpan model dan selector
os.makedirs('model', exist_ok=True)

# Ambil model RandomForest dari pipeline hasil GridSearch
trained_rf_model = grid.best_estimator_.named_steps['rf']

# Simpan model ke model/model_numpy.pkl
with open('model/model_numpy.pkl', 'wb') as f:
    pickle.dump(trained_rf_model, f)

# Simpan selector ke model/selector.pkl
with open('model/selector.pkl', 'wb') as f:
    pickle.dump(selector, f)

print("✅ Model dan selector berhasil disimpan ke folder 'model/'")
