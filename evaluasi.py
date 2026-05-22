import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

def train_rf():
    df_train = pd.read_csv('fruit_train.csv')
    df_test = pd.read_csv('fruit_test.csv')
    df_total = pd.concat([df_train, df_test]) # Gabungkan data
    
    features, labels = [], []
    for _, row in df_total.iterrows():
        img = cv2.imread(row['file'], cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (64, 64))
            features.append(img.flatten())
            labels.append(row['label'])
    
    X = np.array(features)
    y = np.array(labels)

    # PEMBAGIAN DATA 85% Training & 15% Testing[cite: 3]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    print(f"Training RF dengan {len(X_train)} data...")
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("--- HASIL EVALUASI RF (15% Data Testing) ---")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(rf, 'model_fruit_rf.pkl')
    print("✅ Model RF 85:15 tersimpan!")

if __name__ == "__main__":
    train_rf()