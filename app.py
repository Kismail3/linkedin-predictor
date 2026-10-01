import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="LinkedIn User Predictor", page_icon="💼")

@st.cache_data
def load_data():
    s = pd.read_csv("social_media_usage.csv")
    clean = lambda x: np.where(x == 1, 1, 0)
    ss = pd.DataFrame({
        "sm_li": clean(s["web1h"]),
        "income": np.where(s["income"] > 9, np.nan, s["income"]),
        "education": np.where(s["educ2"] > 8, np.nan, s["educ2"]),
        "parent": clean(s["par"]),
        "married": clean(s["marital"]),
        "female": np.where(s["gender"] == 2, 1, 0),
        "age": np.where(s["age"] > 98, np.nan, s["age"]),
    }).dropna()
    return ss

@st.cache_resource
def train_model():
    ss = load_data()
    X, y = ss.drop(columns="sm_li"), ss["sm_li"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=987)
    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)
    return model, X.columns, model.score(X_test, y_test)

model, cols, acc = train_model()
ss = load_data()

st.title("💼 LinkedIn User Predictor")
st.write("Enter a person's demographics to predict whether they use LinkedIn. "
         "Built for the marketing team to size up LinkedIn as a campaign channel.")

income_labels = {1: "1 - Less than $10k", 2: "2 - $10k to <$20k", 3: "3 - $20k to <$30k",
                 4: "4 - $30k to <$40k", 5: "5 - $40k to <$50k", 6: "6 - $50k to <$75k",
                 7: "7 - $75k to <$100k", 8: "8 - $100k to <$150k", 9: "9 - $150k+"}
educ_labels = {1: "1 - Less than high school", 2: "2 - High school incomplete",
               3: "3 - High school graduate", 4: "4 - Some college, no degree",
               5: "5 - Two-year associate degree", 6: "6 - Four-year bachelor's degree",
               7: "7 - Some postgraduate", 8: "8 - Postgraduate degree"}

with st.sidebar:
    st.header("Person's attributes")
    income = st.selectbox("Income", list(income_labels), index=7, format_func=income_labels.get)
    education = st.selectbox("Education", list(educ_labels), index=6, format_func=educ_labels.get)
    age = st.slider("Age", 18, 98, 42)
    parent = st.radio("Parent?", ["No", "Yes"], horizontal=True)
    married = st.radio("Married?", ["No", "Yes"], horizontal=True, index=1)
    female = st.radio("Gender", ["Male", "Female"], horizontal=True, index=1)

person = pd.DataFrame([[income, education, int(parent == "Yes"), int(married == "Yes"),
                        int(female == "Female"), age]], columns=cols)
prob = model.predict_proba(person)[0, 1]
label = "LinkedIn user" if prob >= 0.5 else "Not a LinkedIn user"

st.subheader("Prediction")
c1, c2 = st.columns(2)
c1.metric("Classification", label)
c2.metric("Probability of using LinkedIn", f"{prob:.1%}")
st.progress(float(prob))
if prob >= 0.5:
    st.success("This profile is likely on LinkedIn: a good fit for LinkedIn campaigns.")
else:
    st.info("This profile is less likely to be on LinkedIn: consider other platforms.")

# Optional visual: how probability changes with age for this person
st.subheader("How age changes this person's probability")
ages = np.arange(18, 99)
grid = pd.concat([person] * len(ages), ignore_index=True)
grid["age"] = ages
curve = pd.DataFrame({"Probability": model.predict_proba(grid)[:, 1]}, index=pd.Index(ages, name="Age"))
st.line_chart(curve)

with st.expander("About the model"):
    st.write(f"Logistic regression (balanced class weights) trained on {len(ss):,} Pew respondents. "
             f"Test accuracy: {acc:.1%}. Data are for educational use only.")
