import streamlit as st
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
import os

st.title("Parquet Dataset Inspector")

repo_id = st.text_input(
    "Hugging Face Dataset ID",
    "index-oss/reverseDomain"
)

if st.button("Inspect Dataset"):
    try:
        api = HfApi()
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset"
        )

        parquet_files = [
            f for f in files if f.endswith(".parquet")
        ]

        st.success(f"Parquet files found: {len(parquet_files)}")

        results = []

        for filename in parquet_files:
            info = api.get_paths_info(
                repo_id=repo_id,
                paths=[filename],
                repo_type="dataset"
            )[0]

            size = getattr(info, "size", None)

            results.append({
                "File": filename,
                "Size (GB)": round(size / 1e9, 2)
                if size else None
            })

        st.dataframe(results)

        if st.button("Inspect first file schema"):
            pass

        st.info(
            "File listing complete. Schema inspection "
            "is a separate operation."
        )

    except Exception as e:
        st.error(f"Inspection failed: {e}")
