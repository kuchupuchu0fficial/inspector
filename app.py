import streamlit as st
import pyarrow.parquet as pq
import fsspec
import json
import os
from datetime import datetime
from huggingface_hub import HfApi

st.set_page_config(
    page_title="Complete Parquet Inspector",
    layout="wide"
)

st.title("Complete Parquet Dataset Inspector")
st.caption("Remote metadata inspection — no intentional full-file download")

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

repo_id = st.text_input(
    "Hugging Face Dataset ID",
    value="index-oss/reverseDomain"
)

revision = st.text_input(
    "Dataset revision",
    value="main"
)

max_files = st.number_input(
    "Maximum files to inspect (0 = all)",
    min_value=0,
    value=0,
    step=1
)

inspect_stats = st.checkbox(
    "Inspect row-group column statistics",
    value=True
)

# --------------------------------------------------
# HELPERS
# --------------------------------------------------

def human_size(size):
    if size is None:
        return "Unknown"

    size = float(size)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024


def get_field_info(schema):
    return [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable
        }
        for field in schema
    ]


def safe_stat_value(value):
    if value is None:
        return None

    try:
        return str(value)[:200]
    except Exception:
        return None


# --------------------------------------------------
# INSPECTION
# --------------------------------------------------

if st.button("Inspect Entire Dataset", type="primary"):

    try:
        api = HfApi()

        with st.spinner("Reading Hugging Face file listing..."):
            entries = api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                recursive=True,
                expand=True
            )

            parquet_files = []

            for item in entries:
                path = getattr(item, "path", "")

                if path.lower().endswith(".parquet"):
                    parquet_files.append({
                        "path": path,
                        "size_bytes": getattr(item, "size", None)
                    })

        if not parquet_files:
            st.error("No Parquet files found.")
            st.stop()

        parquet_files.sort(key=lambda item: item["path"])

        total_repo_size = sum(
            item["size_bytes"] or 0
            for item in parquet_files
        )

        st.success(
            f"Found {len(parquet_files)} Parquet files"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Total Parquet Size",
            human_size(total_repo_size)
        )

        c2.metric(
            "Files Found",
            len(parquet_files)
        )

        st.subheader("File Listing")

        file_table = [
            {
                "File": item["path"],
                "Size (bytes)": item["size_bytes"],
                "Size": human_size(item["size_bytes"])
            }
            for item in parquet_files
        ]

        st.dataframe(
            file_table,
            use_container_width=True
        )

        selected_files = parquet_files

        if max_files > 0:
            selected_files = parquet_files[:int(max_files)]

            st.warning(
                f"Inspecting {len(selected_files)} of "
                f"{len(parquet_files)} files."
            )

        # --------------------------------------------------
        # REMOTE METADATA
        # --------------------------------------------------

        st.subheader("Parquet Metadata")

        reports = []
        common_columns = None
        total_known_rows = 0
        files_with_known_rows = 0
        errors = []

        progress = st.progress(0)

        for index, item in enumerate(selected_files):

            filename = item["path"]

            with st.expander(
                f"{index + 1}. {filename}",
                expanded=False
            ):

                report = {
                    "file": filename,
                    "file_size_bytes": item["size_bytes"],
                    "file_size": human_size(item["size_bytes"]),
                    "rows": None,
                    "row_groups": None,
                    "columns": [],
                    "schema": [],
                    "statistics": [],
                    "metadata_error": None
                }

                try:
                    url = (
                        f"https://huggingface.co/datasets/"
                        f"{repo_id}/resolve/{revision}/"
                        f"{filename}"
                    )

                    fs = fsspec.filesystem("http")

                    # Seekable HTTP access is required for remote
                    # Parquet footer access. No intentional full
                    # file download is performed here.
                    with fs.open(url, "rb") as remote_file:

                        parquet = pq.ParquetFile(remote_file)

                        metadata = parquet.metadata
                        schema = parquet.schema_arrow

                        report["rows"] = metadata.num_rows
                        report["row_groups"] = (
                            metadata.num_row_groups
                        )
                        report["columns"] = list(schema.names)
                        report["schema"] = get_field_info(schema)

                        if common_columns is None:
                            common_columns = set(schema.names)

                        else:
                            if set(schema.names) != common_columns:
                                st.warning(
                                    "Schema differs from the first file."
                                )

                        st.write("Rows:", f"{metadata.num_rows:,}")
                        st.write(
                            "Row groups:",
                            metadata.num_row_groups
                        )

                        st.write("Columns and types:")

                        st.dataframe(
                            report["schema"],
                            use_container_width=True
                        )

                        # --------------------------------------------------
                        # ROW GROUP STATISTICS
                        # --------------------------------------------------

                        if inspect_stats:

                            stats_rows = []

                            for rg_index in range(
                                metadata.num_row_groups
                            ):

                                row_group = metadata.row_group(
                                    rg_index
                                )

                                for col_index in range(
                                    row_group.num_columns
                                ):

                                    col = row_group.column(
                                        col_index
                                    )

                                    row = {
                                        "row_group": rg_index,
                                        "column": col.path_in_schema,
                                        "compression": col.compression,
                                        "compressed_bytes": (
                                            col.total_compressed_size
                                        ),
                                        "uncompressed_bytes": (
                                            col.total_uncompressed_size
                                        ),
                                        "has_statistics": (
                                            col.is_stats_set
                                        )
                                    }

                                    if col.is_stats_set:

                                        stats = col.statistics

                                        row["null_count"] = (
                                            stats.null_count
                                        )

                                        row["distinct_count"] = (
                                            stats.distinct_count
                                        )

                                        if stats.has_min_max:
                                            row["min"] = (
                                                safe_stat_value(stats.min)
                                            )
                                            row["max"] = (
                                                safe_stat_value(stats.max)
                                            )

                                    stats_rows.append(row)

                            report["statistics"] = stats_rows

                            st.write("Row-group statistics:")

                            if stats_rows:
                                st.dataframe(
                                    stats_rows,
                                    use_container_width=True
                                )
                            else:
                                st.info(
                                    "No row-group statistics available."
                                )

                    total_known_rows += report["rows"]
                    files_with_known_rows += 1

                except Exception as exc:
                    report["metadata_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

                    errors.append({
                        "file": filename,
                        "error": report["metadata_error"]
                    })

                    st.error(
                        "Could not read metadata: "
                        + report["metadata_error"]
                    )

                reports.append(report)

            progress.progress(
                (index + 1) / len(selected_files)
            )

        # --------------------------------------------------
        # SUMMARY
        # --------------------------------------------------

        st.subheader("Dataset Summary")

        s1, s2, s3 = st.columns(3)

        s1.metric(
            "Files Inspected",
            len(selected_files)
        )

        s2.metric(
            "Files With Row Counts",
            files_with_known_rows
        )

        s3.metric(
            "Known Total Rows",
            f"{total_known_rows:,}"
            if files_with_known_rows == len(selected_files)
            else "Partial"
        )

        if files_with_known_rows != len(selected_files):
            st.info(
                "Total rows are incomplete because one or more "
                "files could not be inspected."
            )

        all_columns = {}

        for report in reports:
            for column in report["schema"]:

                name = column["name"]

                all_columns.setdefault(name, set()).add(
                    column["type"]
                )

        st.subheader("All Discovered Columns")

        column_table = [
            {
                "Column": name,
                "Observed Types": ", ".join(sorted(types))
            }
            for name, types in sorted(all_columns.items())
        ]

        st.dataframe(
            column_table,
            use_container_width=True
        )

        st.subheader("Potential Index File Names")

        candidates = [
            item["path"]
            for item in parquet_files
            if os.path.basename(item["path"]).lower().startswith(
                "idx_"
            )
        ]

        st.write(candidates)

        st.info(
            "A filename beginning with idx_ does not prove that "
            "a database index exists. Parquet row-group statistics "
            "also are not equivalent to a database index."
        )

        # --------------------------------------------------
        # EXPORT
        # --------------------------------------------------

        final_report = {
            "dataset": repo_id,
            "revision": revision,
            "generated_at": datetime.now().isoformat(),
            "total_parquet_files": len(parquet_files),
            "total_parquet_size_bytes": total_repo_size,
            "files_inspected": len(selected_files),
            "files_with_known_rows": files_with_known_rows,
            "total_known_rows": total_known_rows,
            "columns": column_table,
            "files": reports,
            "errors": errors
        }

        st.download_button(
            "Download Complete Metadata JSON",
            data=json.dumps(
                final_report,
                indent=2,
                ensure_ascii=False,
                default=str
            ),
            file_name="dataset_metadata_report.json",
            mime="application/json"
        )

        st.success("Inspection completed.")

    except Exception as exc:
        st.error(
            f"Dataset inspection failed: "
            f"{type(exc).__name__}: {exc}"
        )
