import streamlit as st
import requests
import pyarrow.parquet as pq
import fsspec
import json

from huggingface_hub import HfApi

st.set_page_config(
    page_title="Parquet Inspector",
    layout="wide"
)

st.title("Hugging Face Parquet Inspector")
st.caption("Remote metadata inspection — no intentional full-file download")

repo_id = st.text_input(
    "Dataset repository",
    "index-oss/reverseDomain"
)

if st.button("Inspect Dataset", type="primary"):

    try:
        api = HfApi()

        files = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            recursive=True,
            expand=True
        )

        parquet_files = [
            {
                "path": item.path,
                "size": getattr(item, "size", None)
            }
            for item in files
            if item.path.lower().endswith(".parquet")
        ]

        if not parquet_files:
            st.error("No Parquet files found.")
            st.stop()

        st.success(
            f"Found {len(parquet_files)} Parquet files."
        )

        st.dataframe(
            [
                {
                    "File": item["path"],
                    "Size (GB)": round(
                        item["size"] / 1e9, 2
                    ) if item["size"] else None
                }
                for item in parquet_files
            ],
            use_container_width=True
        )

        selected = st.selectbox(
            "Select a Parquet file to inspect",
            options=[item["path"] for item in parquet_files]
        )

        if st.button("Read Parquet Metadata"):

            with st.spinner(
                "Reading remote Parquet metadata..."
            ):
                url = (
                    "https://huggingface.co/datasets/"
                    f"{repo_id}/resolve/main/{selected}"
                )

                fs = fsspec.filesystem("http")

                with fs.open(url, "rb") as remote_file:
                    parquet = pq.ParquetFile(remote_file)

                    metadata = parquet.metadata
                    schema = parquet.schema_arrow

                    st.subheader("File Information")

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Rows",
                        f"{metadata.num_rows:,}"
                    )

                    c2.metric(
                        "Row Groups",
                        f"{metadata.num_row_groups:,}"
                    )

                    c3.metric(
                        "Columns",
                        f"{len(schema.names):,}"
                    )

                    st.subheader("Schema and Columns")

                    schema_rows = []

                    for field in schema:
                        schema_rows.append({
                            "Column": field.name,
                            "Data Type": str(field.type),
                            "Nullable": field.nullable
                        })

                    st.dataframe(
                        schema_rows,
                        use_container_width=True
                    )

                    st.subheader("Row Group Statistics")

                    stats_rows = []

                    for rg_index in range(
                        metadata.num_row_groups
                    ):
                        rg = metadata.row_group(rg_index)

                        for col_index in range(
                            rg.num_columns
                        ):
                            col = rg.column(col_index)

                            row = {
                                "Row Group": rg_index,
                                "Column": col.path_in_schema,
                                "Compression": col.compression,
                                "Compressed Bytes": (
                                    col.total_compressed_size
                                ),
                                "Uncompressed Bytes": (
                                    col.total_uncompressed_size
                                ),
                                "Statistics Available": (
                                    col.is_stats_set
                                )
                            }

                            if col.is_stats_set:
                                stats = col.statistics

                                row["Null Count"] = (
                                    stats.null_count
                                )

                                row["Distinct Count"] = (
                                    stats.distinct_count
                                )

                                row["Min"] = (
                                    str(stats.min)
                                    if stats.has_min_max
                                    else None
                                )

                                row["Max"] = (
                                    str(stats.max)
                                    if stats.has_min_max
                                    else None
                                )

                            stats_rows.append(row)

                    st.dataframe(
                        stats_rows,
                        use_container_width=True
                    )

                    report = {
                        "dataset": repo_id,
                        "file": selected,
                        "rows": metadata.num_rows,
                        "row_groups": metadata.num_row_groups,
                        "columns": schema_rows,
                        "statistics": stats_rows
                    }

                    st.download_button(
                        "Download Metadata JSON",
                        data=json.dumps(
                            report,
                            indent=2,
                            default=str
                        ),
                        file_name="parquet_metadata.json",
                        mime="application/json"
                    )

                    st.info(
                        "Parquet row-group statistics are not "
                        "the same as a database index. "
                        "This report does not prove that a "
                        "phone-number lookup index exists."
                    )

    except Exception as exc:
        st.error(
            f"Inspection failed: {type(exc).__name__}: {exc}"
        )

        st.warning(
            "Remote metadata access may fail if the server "
            "does not support HTTP range requests, the file "
            "requires authentication, or the hosting "
            "environment cannot access the remote file."
        )
