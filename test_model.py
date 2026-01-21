import joblib
import os

def test_model():
    print("Testing model artifacts...")
    if not os.path.exists('spam_model.pkl') or not os.path.exists('vectorizer.pkl'):
        print("FAIL: Artifacts not found.")
        return

    try:
        model = joblib.load('spam_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        test_text = ["Free money, click here!", "Hello, how are you today?"]
        X_test = vectorizer.transform(test_text)
        probs = model.predict_proba(X_test)
        preds = model.predict(X_test)
        
        print(f"Predictions: {preds} (1=spam, 0=ham)")
        print(f"Probabilities: {probs}")
        print("SUCCESS: Model loaded and predicted.")
    except Exception as e:
        print(f"FAIL: Error during testing: {e}")

if __name__ == "__main__":
    test_model()
