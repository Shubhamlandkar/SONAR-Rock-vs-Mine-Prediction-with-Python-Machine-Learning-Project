import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

st.set_page_config(page_title="Sonar Rock vs Mine", layout="wide")

st.title("Sonar: Rock vs Mine — Interactive ML App")
st.write("Upload the Sonar dataset (CSV) or use sample data. The app trains multiple classifiers and shows performance metrics and confusion matrices.")

# Sidebar - data upload and model selection
st.sidebar.header("Upload / Settings")
uploaded_file = st.sidebar.file_uploader("Upload Sonar CSV file", type=["csv"]) 
use_example = st.sidebar.checkbox("Use example Sonar dataset (if no upload)", value=True)

test_size = st.sidebar.slider("Test set proportion", 0.1, 0.5, 0.25, 0.05)
random_state = st.sidebar.number_input("Random seed", min_value=0, value=23, step=1)
scale_data = st.sidebar.checkbox("Scale features (StandardScaler)", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Choose Algorithms to Train")
use_logreg = st.sidebar.checkbox("Logistic Regression", value=True)
use_knn = st.sidebar.checkbox("K-Nearest Neighbors", value=True)
use_svm = st.sidebar.checkbox("SVM", value=True)
use_rf = st.sidebar.checkbox("Random Forest", value=True)

knn_k = st.sidebar.slider("K for KNN", 1, 25, 5)
rf_estimators = st.sidebar.slider("RF estimators", 10, 500, 100, step=10)

# Load data
@st.cache_data
def load_example():
    # The Sonar dataset has 60 numeric attributes and 'R' or 'M' label in the last column.
    url = "https://raw.githubusercontent.com/plotly/datasets/master/sonar.csv"
    df = pd.read_csv(url, header=None)
    # Some public variants put header; ensure last col name
    df = df.rename(columns={60: 'Label'})
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully")
else:
    if use_example:
        df = load_example()
        st.info("Using example Sonar dataset")
    else:
        st.warning("No dataset provided. Please upload a CSV or enable example dataset.")
        st.stop()

st.write("### Data preview")
st.dataframe(df.head())

# Prepare features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Encode labels if necessary
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_enc = le.fit_transform(y)  # 'R'/'M' -> 0/1

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=int(random_state))

# Scaling
if scale_data:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
else:
    scaler = None

results = {}
models = {}

# Helper to train and evaluate
def train_and_eval(clf, name):
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)
    report = classification_report(y_test, y_pred_test, target_names=le.classes_, output_dict=True)
    results[name] = {
        'model': clf,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'confusion_matrix': cm,
        'report': report
    }
    models[name] = clf

# Train selected models
if use_logreg:
    clf = LogisticRegression(max_iter=1000, random_state=int(random_state))
    train_and_eval(clf, 'Logistic Regression')

if use_knn:
    clf = KNeighborsClassifier(n_neighbors=int(knn_k))
    train_and_eval(clf, 'KNN')

if use_svm:
    clf = SVC(probability=True, random_state=int(random_state))
    train_and_eval(clf, 'SVM')

if use_rf:
    clf = RandomForestClassifier(n_estimators=int(rf_estimators), random_state=int(random_state))
    train_and_eval(clf, 'Random Forest')

# Display results
st.write("## Model performance")
for name, res in results.items():
    st.subheader(name)
    st.write(f"Training accuracy: **{res['train_acc']:.2f}**")
    st.write(f"Testing accuracy: **{res['test_acc']:.2f}**")
    st.write("Confusion matrix:")
    cm = res['confusion_matrix']
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    st.table(cm_df)
    st.write("Classification report:")
    report_df = pd.DataFrame(res['report']).transpose()
    st.dataframe(report_df)

# Insights text
st.markdown("---")
st.header("Project Insight")
st.write("This project demonstrates how machine learning can distinguish between metallic mines and natural rocks based on sonar frequency data. With preprocessing, proper scaling, and model tuning, we can achieve ~70–85% accuracy. Logistic Regression provides a stable interpretable baseline, KNN can give slightly higher accuracy depending on k and scaling, SVM performs well on small datasets, and Random Forest helps identify feature importance and avoid overfitting.")

# Save a selected model as pickle
st.sidebar.header("Export Model")
save_model_name = st.sidebar.selectbox("Choose model to pickle", options=list(models.keys()) if models else [])
pickle_filename = st.sidebar.text_input("Pickle filename", value="sonar_model.pkl")

if st.sidebar.button("Save model as pickle"):
    if not save_model_name:
        st.sidebar.error("No model available to save")
    else:
        chosen = models[save_model_name]
        # Save scaler + model together
        artifact = {'scaler': scaler, 'model': chosen, 'label_encoder': le}
        with open(pickle_filename, 'wb') as f:
            pickle.dump(artifact, f)
        st.sidebar.success(f"Saved {save_model_name} to {pickle_filename}")
        st.markdown(f"[Download the pickle file](./{pickle_filename})")

# Single prediction
st.sidebar.header("Single Prediction")
input_vals = []
if st.sidebar.button("Show input fields for single prediction"):
    with st.form("single_pred_form"):
        st.write("Provide 60 feature values (float). You can paste comma-separated values.")
        text = st.text_area("Paste 60 comma-separated numeric features", height=120)
        selected_model = st.selectbox("Choose model for prediction", options=list(models.keys()) if models else [])
        submit = st.form_submit_button("Predict")
        if submit:
            if not text:
                st.error("Please paste feature vector")
            else:
                try:
                    vals = np.array([float(x.strip()) for x in text.split(',')])
                    if vals.size != X.shape[1]:
                        st.error(f"Expected {X.shape[1]} features, got {vals.size}")
                    else:
                        if scale_data and scaler is not None:
                            vals = scaler.transform(vals.reshape(1, -1))
                        pred = models[selected_model].predict(vals.reshape(1, -1))
                        label = le.inverse_transform(pred)[0]
                        st.success(f"Predicted class: {label}")
                except Exception as e:
                    st.error(f"Error parsing input: {e}")

st.write("---")
st.caption("Built with ❤️ — modify hyperparameters or upload your own Sonar dataset CSV to experiment")
