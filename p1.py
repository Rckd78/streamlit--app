
#pip3 install streamlit pandas google-generative
import streamlit as st
import pandas as pd
import google.generativeai as genai

api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

model = genai.GenerativeModel("gemini-2.5-flash") 

st.title("🔍 Product Feature Extractor")

st.markdown("""
This app extracts key product features (e.g., **battery life**, **camera quality**) from customer reviews using prompt engineering and foundation models.
Upload your product review dataset below and get a summary of the most and least appreciated aspects.
""")

uploaded_file = st.file_uploader("📄 Upload your product review CSV file", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        if 'review' not in df.columns:
            st.error("❌ The CSV must contain a 'review' column.")
        else:
            st.success("✅ File uploaded successfully!")
            st.write("Sample data preview:")
            st.dataframe(df.head())

            if st.button("🧠 Extract Product Features"):
                with st.spinner("Extracting features from reviews..."):
                    extracted_features = []

                    for review in df['review']:
                        prompt = (
                            f"Extract product features mentioned in the review below. "
                            f"Focus only on specific aspects like 'battery life', 'camera quality', 'display', etc.:\n\n"
                            f"Review: \"{review}\"\n\n"
                            f"List key product features only."
                        )

                        try:
                            response = model.generate_content(prompt)
                            features = response.text.strip()
                        except Exception as e:
                            features = f"Error: {e}"

                        extracted_features.append(features)
                    df['Extracted_Features'] = extracted_features

                    st.success("✅ Feature extraction complete!")
                    st.dataframe(df[['review', 'Extracted_Features']])

                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download Results as CSV", csv, "extracted_features.csv", "text/csv")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")
else:
    st.info("📌 Please upload a CSV file with a 'review' column.")
