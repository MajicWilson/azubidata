import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv('bank-additional-full.csv', sep=';')

# Drop redundant features based on correlation analysis
data = data.drop(columns=['emp.var.rate', 'nr.employed'])

# Encode the target variable
X = data.drop(columns=['y'])
y = data['y'].map({'no': 0, 'yes': 1})

# Preprocessing: Scaling numerical features and encoding categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

X_transformed = preprocessor.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.3, random_state=42, stratify=y)

# Train a Random Forest model
model = RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predict and evaluate
threshold = 0.3
y_pred = (model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
print(classification_report(y_test, y_pred))
