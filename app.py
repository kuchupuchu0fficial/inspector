import streamlit as st
import requests

st.title("Parquet Schema Inspector")

repo = "index-oss/reverseDomain"

files = [
    "idx_aadhar.4.parquet",
    "idx_aadhar.5.parquet",
    "idx_aadhar.6.parquet",
    "idx_phone.0.parquet",
]

for filename in files:
    if st.button(f"Inspect {filename}"):
        url = (
            f"https://huggingface.co/datasets/"
            f"{repo}/resolve/main/{filename}"
        )

        st.write("File:", filename)

        try:
            response = requests.head(
                url,
                allow_redirects=True,
                timeout=30
            )

            st.write(
                "HTTP status:",
                response.status_code
            )

            st.write(
                "Content length:",
                response.headers.get("Content-Length")
            )

            st.info(
                "This checks file availability only. "
                "It does not retrieve the Parquet schema."
            )

        except requests.RequestException as error:
            st.error(str(error))
