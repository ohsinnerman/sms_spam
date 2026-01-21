# SMS Spam Detection App

This is a Streamlit web application that detects whether an SMS message is **Spam** or **Ham** (Legitimate) using a Multinomial Naive Bayes model trained on the SMS Spam Collection dataset.

## Setup & Installation

1.  **Install Dependencies**
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Generate Model Artifacts**
    The app requires pre-trained model files. If `spam_model.pkl` and `vectorizer.pkl` are not present, generate them by running:
    ```bash
    python generate_artifacts.py
    ```
    *(Note: This requires `spam.csv` and `combined_dataset.csv` to be in the project directory)*

## Running the App

Run the following command to start the Streamlit server:
```bash
streamlit run app.py
```

## Features

- **Real-time Classification**: Predicts Spam/Ham instantly.
- **Confidence Score**: Shows the probability of the prediction.
- **Adjustable Threshold**: Fine-tune the sensitivity in the sidebar.

## Disclaimer

This model is trained on a specific historical dataset and may not generalize perfectly to all modern spam trends. It is intended for educational and demonstration purposes.
