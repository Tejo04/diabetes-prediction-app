import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# -------------------------------------
# Load and clean data
# -------------------------------------
print("🔄 Loading dataset...")
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv")

# Replace zeros with median in relevant columns
cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_fix:
    df[col] = df[col].replace(0, df[col].median())

# Features & target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# -------------------------------------
# Train-test split + SMOTE
# -------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

print("🔁 Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# -------------------------------------
# Scaling
# -------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)

# -------------------------------------
# Model training
# -------------------------------------
print("🧠 Training models...")

models = {
    "logreg": LogisticRegression(max_iter=1000),
    "svm": SVC(probability=True),
    "knn": KNeighborsClassifier(),
    "dt": DecisionTreeClassifier(),
    "rf": RandomForestClassifier(random_state=42),
    "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train and save each model
for name, model in models.items():
    print(f"✅ Training {name.upper()}...")
    model.fit(X_train_scaled, y_train_resampled)
    joblib.dump(model, f"{name}_model.pkl")

# Save the scaler
joblib.dump(scaler, "scaler.pkl")

print("\n✅ All models and scaler saved successfully!")
