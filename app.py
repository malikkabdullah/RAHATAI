import streamlit as st

# ------------------------------
# BASIC PAGE SETTINGS
# ------------------------------
st.set_page_config(page_title="Simple Demo App", page_icon="⚡", layout="centered")

# ------------------------------
# TITLE & DESCRIPTION
# ------------------------------
st.title("⚡ Streamlit UI Demo")
st.write("This is a very simple example to understand how Streamlit works.")

# ------------------------------
# TEXT INPUT
# ------------------------------
st.subheader("🔹 Enter Some Text")
user_text = st.text_input("Type anything here:")

# ------------------------------
# SLIDER INPUT
# ------------------------------
st.subheader("🔹 Select a Number")
number = st.slider("Choose a value:", 1, 100, 50)

# ------------------------------
# DROPDOWN SELECT BOX
# ------------------------------
st.subheader("🔹 Choose a Model (just for demo)")
model_choice = st.selectbox(
    "Select a model:",
    ["Naive Bayes", "SVM", "LSTM", "CNN", "Transformer"]
)

# ------------------------------
# BUTTON ACTION
# ------------------------------
st.subheader("🔹 Actions")

if st.button("Show Output"):
    st.success("Button clicked successfully!")
    st.write("📌 Your Text:", user_text)
    st.write("📌 Selected Number:", number)
    st.write("📌 Selected Model:", model_choice)

# ------------------------------
# EXPANDER
# ------------------------------
with st.expander("Click to see more info"):
    st.write("This is extra information inside an expander box.")

# ------------------------------
# COLUMNS EXAMPLE
# ------------------------------
st.subheader("🔹 Columns Example")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("Column 1")
with col2:
    st.warning("Column 2")
with col3:
    st.error("Column 3")

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.write("🎉 This is just a UI demo. No models included.")
