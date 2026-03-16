import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="SEC Filing Impact Prediction", layout="wide")

st.title("SEC Filing Impact Prediction Dashboard")
st.markdown(
    """
    This dashboard displays prediction results generated in RapidMiner
    for hypothetical SEC filing scenarios.
    """
)

# Load CSV
df = pd.read_csv("prediction_output.csv")

st.subheader("Prediction Table")
st.dataframe(df)

# Select row
row_index = st.selectbox(
    "Choose example scenario",
    df.index,
    format_func=lambda x: f"Scenario {x + 1}"
)

selected = df.loc[row_index]

prediction = selected["prediction(label)"]
conf_false = selected["confidence(false)"]
conf_true = selected["confidence(true)"]

st.subheader("Model Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Prediction", str(prediction).upper())

with col2:
    st.metric("Confidence TRUE", f"{conf_true:.3f}")

with col3:
    st.metric("Confidence FALSE", f"{conf_false:.3f}")

if str(prediction).lower() == "true":
    st.success("Prediction: TRUE - stock price likely increases within 5 days after filing.")
else:
    st.error("Prediction: FALSE - stock price likely does not increase within 5 days after filing.")

st.subheader("Selected Input Features")

feature_df = pd.DataFrame({
    "Feature": ["Assets", "Liabilities", "NetIncomeLoss", "OperatingIncomeLoss", "CPI"],
    "Value": [
        selected["Assets"],
        selected["Liabilities"],
        selected["NetIncomeLoss"],
        selected["OperatingIncomeLoss"],
        selected["cpi"]
    ]
})

st.table(feature_df)

st.subheader("Prediction Confidence Visualization")

labels = ["False", "True"]
values = [conf_false, conf_true]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(labels, values)

ax.set_ylabel("Probability")
ax.set_title("Prediction Confidence")
ax.set_ylim(0, 1)

for bar, value in zip(bars, values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

st.pyplot(fig)