import io
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


st.set_page_config(page_title="Learning Reflection Dashboard", page_icon="📊", layout="wide")

TOPIC_ORDER = ["Cells", "Movement of Substances", "Biological Molecule", "WA1"]
TOPIC_ALIASES = {
    "cells": "Cells",
    "movement of substances": "Movement of Substances",
    "movement of substance": "Movement of Substances",
    "biological molecule": "Biological Molecule",
    "biological molecules": "Biological Molecule",
    "wa1": "WA1",
    "weighted assessment 1": "WA1",
}


# -----------------------------
# Helpers
# -----------------------------
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    replacements = {
        "\xa0": " ",
        "â€“": "–",
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split()).strip()


def split_multi(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []
    separators = [";", "|", ","]
    parts = [text]
    for sep in separators:
        if sep in text:
            parts = [p.strip() for p in text.split(sep)]
            break
    return [p for p in parts if p]


def normalise_topic(value: str) -> str:
    low = clean_text(value).lower()
    return TOPIC_ALIASES.get(low, clean_text(value))


def find_col(columns: List[str], contains: List[str] = None, exact: str = None) -> str:
    clean_map = {c: clean_text(c).lower() for c in columns}
    if exact:
        target = clean_text(exact).lower()
        for original, low in clean_map.items():
            if low == target:
                return original
    if contains:
        for original, low in clean_map.items():
            if all(token.lower() in low for token in contains):
                return original
    raise KeyError(f"Unable to find column with exact={exact} or contains={contains}")


def detect_optional_class_col(columns: List[str]) -> Optional[str]:
    candidates = [
        ["class"],
        ["form", "class"],
        ["teaching", "group"],
        ["group"],
    ]
    for cand in candidates:
        try:
            return find_col(columns, contains=cand)
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(uploaded_file)
    else:
        excel = pd.ExcelFile(uploaded_file)
        raw = pd.read_excel(uploaded_file, sheet_name=excel.sheet_names[0])
    return prepare_df(raw)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]
    cols = df.columns.tolist()

    username_col = find_col(cols, contains=["username"])
    marks_col = find_col(cols, contains=["marks", "current", "topical", "quiz"])
    prep_col = find_col(cols, contains=["preparation", "strategies"])
    skill_col = find_col(cols, contains=["learning", "skills"])
    issue_col = find_col(cols, contains=["main", "issues"])
    next_action_col = find_col(cols, contains=["one thing", "differently", "next time"])

    topic_col = None
    for candidate in (["topic"], ["assessment"], ["s"]):
        try:
            if candidate == ["s"]:
                topic_col = find_col(cols, exact="s")
            else:
                topic_col = find_col(cols, contains=candidate)
            break
        except Exception:
            continue
    if topic_col is None:
        raise KeyError("Could not find a Topic / Assessment column.")

    class_col = detect_optional_class_col(cols)

    reflection_candidates = []
    for c in cols:
        low = clean_text(c).lower()
        if c in [username_col, marks_col, prep_col, skill_col, issue_col, next_action_col, topic_col]:
            continue
        if "reflect" in low or "expectation" in low or "strategy has changed" in low:
            reflection_candidates.append(c)

    rename_map = {
        username_col: "Username",
        marks_col: "Marks",
        prep_col: "Preparation Strategies",
        skill_col: "Learning Skills",
        issue_col: "Performance Issues",
        next_action_col: "Next Time Action",
        topic_col: "Topic",
    }
    if class_col:
        rename_map[class_col] = "Class"

    df = df.rename(columns=rename_map)

    for i, col in enumerate(reflection_candidates, start=1):
        if col not in rename_map:
            df = df.rename(columns={col: f"Reflection {i}"})

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].map(clean_text)

    df["Username"] = df["Username"].astype(str)
    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")
    df["Topic"] = df["Topic"].map(normalise_topic)

    if "Class" not in df.columns:
        df["Class"] = "All Students"
    else:
        df["Class"] = df["Class"].replace("", "Unspecified")

    keep_cols = [
        "Username",
        "Class",
        "Topic",
        "Marks",
        "Preparation Strategies",
        "Learning Skills",
        "Performance Issues",
        "Next Time Action",
    ] + [c for c in df.columns if c.startswith("Reflection ")]

    df = df[keep_cols].copy()
    df = df.dropna(subset=["Username", "Topic", "Marks"])
    df = df[df["Topic"].isin(TOPIC_ORDER)].copy()
    df["Topic"] = pd.Categorical(df["Topic"], categories=TOPIC_ORDER, ordered=True)
    df = df.sort_values(["Class", "Username", "Topic"]).reset_index(drop=True)
    return df


def explode_categories(df: pd.DataFrame, source_col: str, output_col: str = "Category") -> pd.DataFrame:
    temp = df[["Username", "Class", "Topic", "Marks", source_col]].copy()
    temp[output_col] = temp[source_col].apply(split_multi)
    temp = temp.explode(output_col)
    temp[output_col] = temp[output_col].fillna("No response")
    temp = temp[temp[output_col] != ""]
    return temp


def filter_df(df: pd.DataFrame, classes: List[str], topics: List[str], students: List[str]) -> pd.DataFrame:
    out = df.copy()
    if classes:
        out = out[out["Class"].isin(classes)]
    if topics:
        out = out[out["Topic"].astype(str).isin(topics)]
    if students:
        out = out[out["Username"].isin(students)]
    return out


# -----------------------------
# Charts
# -----------------------------
def plot_histogram(df: pd.DataFrame, nbins: int, colour_by_topic: bool) -> go.Figure:
    fig = px.histogram(
        df,
        x="Marks",
        color="Topic" if colour_by_topic else None,
        nbins=nbins,
        marginal="box",
        barmode="overlay" if colour_by_topic else "relative",
        opacity=0.78,
        hover_data=["Username", "Class", "Topic"],
        title="Class Distribution of Scores",
    )
    fig.update_layout(height=500, bargap=0.06, legend_title_text="Topic")
    fig.update_xaxes(title="Marks")
    fig.update_yaxes(title="Number of responses")
    return fig


def plot_topic_sankey(df: pd.DataFrame, source_col: str, title: str) -> go.Figure:
    ex = explode_categories(df, source_col, "Choice")
    counts = ex.groupby(["Topic", "Choice"], observed=True).size().reset_index(name="Count")

    topic_nodes = [t for t in TOPIC_ORDER if t in counts["Topic"].astype(str).tolist()]
    choice_nodes = counts["Choice"].drop_duplicates().astype(str).tolist()
    labels = topic_nodes + choice_nodes
    idx = {label: i for i, label in enumerate(labels)}

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=20,
                    line=dict(color="rgba(90,90,90,0.35)", width=0.5),
                    label=labels,
                ),
                link=dict(
                    source=[idx[str(t)] for t in counts["Topic"].astype(str)],
                    target=[idx[str(c)] for c in counts["Choice"].astype(str)],
                    value=counts["Count"].tolist(),
                    customdata=[
                        f"{t} → {c}<br>Count: {n}"
                        for t, c, n in zip(counts["Topic"], counts["Choice"], counts["Count"])
                    ],
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(title=title, height=620, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def plot_scatter(df: pd.DataFrame, category_col: str) -> go.Figure:
    ex = explode_categories(df, category_col, "Category")
    if ex.empty:
        return go.Figure()

    category_order = ex["Category"].drop_duplicates().tolist()
    cat_map = {cat: i for i, cat in enumerate(category_order)}
    rng = np.random.default_rng(42)
    ex["Y"] = ex["Category"].map(cat_map) + rng.normal(0, 0.08, len(ex))

    fig = px.scatter(
        ex,
        x="Marks",
        y="Y",
        color="Topic",
        symbol="Topic",
        hover_name="Username",
        hover_data={
            "Class": True,
            "Category": True,
            "Marks": True,
            "Topic": True,
            "Y": False,
        },
        title=f"Marks vs {category_col}",
    )
    fig.update_layout(height=max(460, len(category_order) * 34 + 180), legend_title_text="Topic")
    fig.update_xaxes(title="Marks")
    fig.update_yaxes(
        title="Category",
        tickmode="array",
        tickvals=list(cat_map.values()),
        ticktext=list(cat_map.keys()),
    )
    return fig


def plot_student_line(student_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        student_df.sort_values("Topic"),
        x="Topic",
        y="Marks",
        markers=True,
        text="Marks",
        title="Marks Over Time",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=360)
    fig.update_xaxes(title="Topic")
    fig.update_yaxes(title="Marks")
    return fig


def plot_student_skill_sankey(student_df: pd.DataFrame) -> go.Figure:
    return plot_topic_sankey(student_df, "Learning Skills", "Learning Skills Across Topics")


# -----------------------------
# PDF
# -----------------------------
def fig_to_png(fig, width=1200, height=650, scale=2) -> Optional[str]:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        fig.write_image(tmp.name, width=width, height=height, scale=scale)
        return tmp.name
    except Exception:
        return None


def student_action_table(student_df: pd.DataFrame) -> pd.DataFrame:
    out = student_df[["Topic", "Next Time Action"]].copy()
    out["Topic"] = out["Topic"].astype(str)
    return out


def build_student_pdf(student_df: pd.DataFrame, username: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Compact", fontSize=9, leading=12))
    elements = []

    elements.append(Paragraph("Individual Learning Progress Report", styles["Title"]))
    elements.append(Paragraph(f"Student: {username}", styles["Heading2"]))
    if "Class" in student_df.columns:
        elements.append(Paragraph(f"Class: {student_df['Class'].iloc[0]}", styles["BodyText"]))
    elements.append(Spacer(1, 0.16 * inch))

    reflection_cols = [c for c in student_df.columns if c.startswith("Reflection ")]
    elements.append(Paragraph("Reflections", styles["Heading3"]))
    if reflection_cols:
        for _, row in student_df.sort_values("Topic").iterrows():
            elements.append(Paragraph(f"<b>{row['Topic']}</b>", styles["BodyText"]))
            for col in reflection_cols:
                text = clean_text(row.get(col, ""))
                if text:
                    elements.append(Paragraph(text, styles["Compact"]))
            elements.append(Spacer(1, 0.06 * inch))
    else:
        elements.append(Paragraph("No reflection columns were detected in the uploaded file.", styles["BodyText"]))

    line_img = fig_to_png(plot_student_line(student_df), width=1200, height=480)
    sankey_img = fig_to_png(plot_student_skill_sankey(student_df), width=1200, height=700)

    if line_img:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph("Marks Over Topics", styles["Heading3"]))
        elements.append(Image(line_img, width=7.0 * inch, height=2.9 * inch))
    if sankey_img:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph("Learning Skills Sankey", styles["Heading3"]))
        elements.append(Image(sankey_img, width=7.0 * inch, height=4.0 * inch))
    if not line_img and not sankey_img:
        elements.append(Paragraph("Chart images could not be embedded. Install kaleido to enable plot export.", styles["BodyText"]))

    elements.append(Spacer(1, 0.12 * inch))
    elements.append(Paragraph("One Thing I Will Do Differently Next Time", styles["Heading3"]))
    table_df = student_action_table(student_df)
    data = [["Topic", "One thing I will do differently next time"]] + table_df.values.tolist()
    table = Table(data, repeatRows=1, colWidths=[1.8 * inch, 5.2 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3b73")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    elements.append(table)

    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


# -----------------------------
# UI
# -----------------------------
st.title("📊 Learning Reflection Dashboard")
st.caption("Upload your Excel or CSV file to explore score distributions, reflection patterns, and individual student progress.")

with st.sidebar:
    st.header("1) Upload file")
    uploaded = st.file_uploader("Upload Excel / CSV", type=["xlsx", "xls", "csv"])
    st.markdown(
        """
        **Expected fields**
        - Username
        - Marks for Current Topical Quiz
        - Preparation strategies
        - Learning skills
        - Performance issues
        - Topic / Assessment
        - One thing I will do differently next time
        """
    )

if not uploaded:
    st.info("Upload the reflection dataset to begin.")
    st.stop()

try:
    df = load_data(uploaded)
except Exception as e:
    st.error(f"Unable to process file: {e}")
    st.stop()

with st.sidebar:
    st.header("2) Filters")
    class_options = sorted(df["Class"].dropna().unique().tolist())
    topic_options = [t for t in TOPIC_ORDER if t in df["Topic"].astype(str).unique().tolist()]
    student_options = sorted(df["Username"].dropna().unique().tolist())

    selected_classes = st.multiselect("Class filter", class_options, default=class_options)
    selected_topics = st.multiselect("Topic filter", topic_options, default=topic_options)
    selected_students = st.multiselect("Student filter", student_options, default=[])
    st.divider()
    st.header("3) Chart options")
    bins = st.slider("Histogram bins", min_value=5, max_value=30, value=10)
    colour_by_topic = st.toggle("Colour histogram by topic", value=True)

filtered_df = filter_df(df, selected_classes, selected_topics, selected_students)

if filtered_df.empty:
    st.warning("No records match the selected filters.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Students", filtered_df["Username"].nunique())
m2.metric("Responses", len(filtered_df))
m3.metric("Mean marks", f"{filtered_df['Marks'].mean():.1f}")
m4.metric("Median marks", f"{filtered_df['Marks'].median():.1f}")

with st.expander("Preview cleaned data"):
    st.dataframe(filtered_df, use_container_width=True, hide_index=True)

summary_col1, summary_col2 = st.columns([1.2, 1])
with summary_col1:
    topic_summary = (
        filtered_df.groupby("Topic", observed=True)["Marks"]
        .agg(["count", "mean", "median", "min", "max"])
        .reset_index()
    )
    st.markdown("### Topic summary")
    st.dataframe(topic_summary, use_container_width=True, hide_index=True)
with summary_col2:
    st.markdown("### Quick reading")
    highest_topic = topic_summary.sort_values("mean", ascending=False).iloc[0]["Topic"]
    lowest_topic = topic_summary.sort_values("mean", ascending=True).iloc[0]["Topic"]
    st.info(
        f"Highest average: **{highest_topic}**  \n"
        f"Lowest average: **{lowest_topic}**  \n"
        f"Use the histogram and scatter plots below to inspect spread and patterns."
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "Histogram",
    "Preparation Strategies Sankey",
    "Scatter Plot",
    "Individual Student Report",
])

with tab1:
    st.subheader("Histogram of class distribution")
    st.plotly_chart(plot_histogram(filtered_df, bins, colour_by_topic), use_container_width=True)

with tab2:
    st.subheader("Preparation strategies across topics")
    sankey = plot_topic_sankey(
        filtered_df.sort_values("Topic"),
        "Preparation Strategies",
        "Preparation Strategies Used Across Topics",
    )
    st.plotly_chart(sankey, use_container_width=True)
    st.caption("The flow width shows how often a strategy appears within each topic.")

with tab3:
    st.subheader("Marks against learning behaviour / issues")
    scatter_option = st.radio(
        "Choose the categorical variable",
        ["Learning Skills", "Performance Issues"],
        horizontal=True,
    )
    st.plotly_chart(plot_scatter(filtered_df, scatter_option), use_container_width=True)
    st.caption("Each point represents a student response. Hover to see the username, topic, class and marks.")

with tab4:
    st.subheader("Individual student progress")
    report_students = sorted(filtered_df["Username"].unique().tolist())
    selected_student = st.selectbox("Choose a student", report_students)
    student_df = filtered_df[filtered_df["Username"] == selected_student].sort_values("Topic").copy()

    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.plotly_chart(plot_student_line(student_df), use_container_width=True)
    with c2:
        st.plotly_chart(plot_student_skill_sankey(student_df), use_container_width=True)

    st.markdown("### Student reflections")
    reflection_cols = [c for c in student_df.columns if c.startswith("Reflection ")]
    if reflection_cols:
        for _, row in student_df.iterrows():
            st.markdown(f"**{row['Topic']}**")
            for col in reflection_cols:
                value = clean_text(row.get(col, ""))
                if value:
                    st.write(value)
    else:
        st.info("No reflection columns were detected in the uploaded file.")

    st.markdown("### One thing I will do differently next time")
    st.dataframe(student_action_table(student_df), use_container_width=True, hide_index=True)

    pdf_bytes = build_student_pdf(student_df, selected_student)
    safe_name = selected_student.replace("@", "_at_").replace(".", "_").replace(" ", "_")
    st.download_button(
        label="Download student PDF report",
        data=pdf_bytes,
        file_name=f"{safe_name}_learning_progress.pdf",
        mime="application/pdf",
    )

st.markdown("---")
st.caption("Tip: install `kaleido` to embed Plotly charts inside the exported PDF.")
