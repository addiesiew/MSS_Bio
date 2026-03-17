import io
import math
import tempfile
from typing import List, Dict

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

st.set_page_config(page_title="Learning Progress Dashboard", layout="wide")

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


def normalise_text(x):
    if pd.isna(x):
        return ""
    s = str(x)
    s = s.replace("‚Äì", "–").replace("â€“", "–").replace("\xa0", " ")
    return " ".join(s.split()).strip()


def clean_multiselect(text: str) -> List[str]:
    text = normalise_text(text)
    if not text:
        return []
    parts = [p.strip() for p in text.split(";") if p.strip()]
    return parts


def find_col(columns: List[str], contains: List[str], exact: str = None) -> str:
    cleaned = {c: normalise_text(c).lower() for c in columns}
    if exact:
        exact_clean = normalise_text(exact).lower()
        for original, low in cleaned.items():
            if low == exact_clean:
                return original
    for original, low in cleaned.items():
        if all(term.lower() in low for term in contains):
            return original
    raise KeyError(f"Could not find a column matching: {contains}")


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        # use first sheet by default
        df = pd.read_excel(uploaded_file, sheet_name=xls.sheet_names[0])
    return prepare_df(df)


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c) for c in df.columns]

    username_col = find_col(df.columns.tolist(), ["username"])
    marks_col = find_col(df.columns.tolist(), ["marks", "current", "topical", "quiz"])
    prep_col = find_col(df.columns.tolist(), ["preparation", "strategies"])
    skill_col = find_col(df.columns.tolist(), ["learning", "skills"])
    issue_col = find_col(df.columns.tolist(), ["main", "issues"])
    reflect_col = find_col(df.columns.tolist(), ["one thing", "differently", "next time"])
    topic_col = None
    try:
        topic_col = find_col(df.columns.tolist(), ["topic"])
    except Exception:
        topic_col = find_col(df.columns.tolist(), ["assessment"]) if False else None
    if topic_col is None:
        # many uploaded files use short header 's'
        for c in df.columns:
            if normalise_text(c).lower() in {"s", "topic"}:
                topic_col = c
                break
    if topic_col is None:
        raise KeyError("Could not find the Topic column.")

    reflection_cols = [c for c in df.columns if "reflect" in normalise_text(c).lower() or "expectation" in normalise_text(c).lower()]

    rename_map = {
        username_col: "Username",
        marks_col: "Marks",
        prep_col: "Preparation Strategies",
        skill_col: "Learning Skills",
        issue_col: "Performance Issues",
        reflect_col: "Next Time Action",
        topic_col: "Topic",
    }
    df = df.rename(columns=rename_map)

    # Optional narrative reflection columns
    if reflection_cols:
        for idx, c in enumerate(reflection_cols, start=1):
            if c not in rename_map:
                df = df.rename(columns={c: f"Reflection {idx}"})

    df["Username"] = df["Username"].map(normalise_text)
    df["Topic"] = df["Topic"].map(lambda x: TOPIC_ALIASES.get(normalise_text(x).lower(), normalise_text(x)))
    df["Marks"] = pd.to_numeric(df["Marks"], errors="coerce")
    df["Preparation Strategies"] = df["Preparation Strategies"].map(normalise_text)
    df["Learning Skills"] = df["Learning Skills"].map(normalise_text)
    df["Performance Issues"] = df["Performance Issues"].map(normalise_text)
    df["Next Time Action"] = df["Next Time Action"].map(normalise_text)

    for col in [c for c in df.columns if c.startswith("Reflection ")]:
        df[col] = df[col].map(normalise_text)

    df = df.dropna(subset=["Username", "Topic", "Marks"]).copy()
    df["Topic"] = pd.Categorical(df["Topic"], categories=TOPIC_ORDER, ordered=True)
    df = df.sort_values(["Username", "Topic"]).reset_index(drop=True)
    return df


def explode_multiselect(df: pd.DataFrame, col_name: str, out_col: str) -> pd.DataFrame:
    tmp = df[["Username", "Topic", "Marks", col_name]].copy()
    tmp[out_col] = tmp[col_name].apply(clean_multiselect)
    tmp = tmp.explode(out_col)
    tmp[out_col] = tmp[out_col].fillna("No response")
    tmp = tmp[tmp[out_col] != ""]
    return tmp


def build_topic_strategy_sankey(df: pd.DataFrame, column_name: str, title: str) -> go.Figure:
    ex = explode_multiselect(df, column_name, "Item")
    counts = ex.groupby(["Topic", "Item"], observed=True).size().reset_index(name="Count")

    topic_nodes = [t for t in TOPIC_ORDER if t in counts["Topic"].astype(str).tolist()]
    strategy_nodes = counts["Item"].dropna().astype(str).drop_duplicates().tolist()
    labels = topic_nodes + strategy_nodes
    node_index = {label: i for i, label in enumerate(labels)}

    sources = [node_index[str(t)] for t in counts["Topic"].astype(str)]
    targets = [node_index[str(i)] for i in counts["Item"].astype(str)]
    values = counts["Count"].tolist()
    hover = [f"{t} → {i}<br>Count: {c}" for t, i, c in zip(counts["Topic"], counts["Item"], counts["Count"])]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=20,
                    line=dict(color="rgba(80,80,80,0.4)", width=0.5),
                    label=labels,
                ),
                link=dict(source=sources, target=targets, value=values, customdata=hover, hovertemplate="%{customdata}<extra></extra>"),
            )
        ]
    )
    fig.update_layout(title=title, height=620, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def build_scatter(df: pd.DataFrame, y_col: str, chart_title: str) -> go.Figure:
    ex = explode_multiselect(df, y_col, "Category")
    # jitter for visibility
    rng = np.random.default_rng(42)
    ex["Jitter"] = rng.normal(0, 0.08, len(ex))
    cats = ex["Category"].dropna().unique().tolist()
    cat_map = {cat: i for i, cat in enumerate(cats)}
    ex["YPos"] = ex["Category"].map(cat_map) + ex["Jitter"]

    fig = px.scatter(
        ex,
        x="Marks",
        y="YPos",
        color="Topic",
        hover_name="Username",
        hover_data={"Category": True, "Topic": True, "Marks": True, "YPos": False, "Jitter": False},
        title=chart_title,
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(cat_map.values()),
        ticktext=list(cat_map.keys()),
        title="Category",
    )
    fig.update_xaxes(title="Marks")
    fig.update_layout(height=max(450, 32 * len(cats) + 180), legend_title_text="Topic")
    return fig


def build_student_line(student_df: pd.DataFrame) -> go.Figure:
    fig = px.line(
        student_df.sort_values("Topic"),
        x="Topic",
        y="Marks",
        markers=True,
        title="Marks over Topics",
        text="Marks",
    )
    fig.update_traces(textposition="top center")
    fig.update_layout(height=380)
    return fig


def build_student_skill_sankey(student_df: pd.DataFrame) -> go.Figure:
    return build_topic_strategy_sankey(student_df, "Learning Skills", "Student Learning Skills by Topic")


def fig_to_temp_png(fig, width=1200, height=700, scale=2) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.write_image(tmp.name, width=width, height=height, scale=scale)
    return tmp.name


def student_reflection_table(student_df: pd.DataFrame) -> pd.DataFrame:
    out = student_df[["Topic", "Next Time Action"]].copy()
    out["Topic"] = out["Topic"].astype(str)
    return out


def build_student_pdf(student_df: pd.DataFrame, username: str) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallWrap", fontSize=9, leading=12))
    elements = []

    elements.append(Paragraph(f"Individual Learning Progress Report", styles["Title"]))
    elements.append(Paragraph(f"Student: {username}", styles["Heading2"]))
    elements.append(Spacer(1, 0.18 * inch))

    # Narrative reflections
    reflection_cols = [c for c in student_df.columns if c.startswith("Reflection ")]
    if reflection_cols:
        elements.append(Paragraph("Reflections", styles["Heading3"]))
        for _, row in student_df.sort_values("Topic").iterrows():
            topic = str(row["Topic"])
            elements.append(Paragraph(f"<b>{topic}</b>", styles["BodyText"]))
            for col in reflection_cols:
                val = row.get(col, "")
                if normalise_text(val):
                    elements.append(Paragraph(normalise_text(val), styles["SmallWrap"]))
            elements.append(Spacer(1, 0.08 * inch))

    # Charts (requires kaleido)
    try:
        line_path = fig_to_temp_png(build_student_line(student_df), width=1200, height=500)
        sankey_path = fig_to_temp_png(build_student_skill_sankey(student_df), width=1200, height=680)
        elements.append(Spacer(1, 0.12 * inch))
        elements.append(Paragraph("Marks over Topics", styles["Heading3"]))
        elements.append(Image(line_path, width=7.0 * inch, height=3.0 * inch))
        elements.append(Spacer(1, 0.15 * inch))
        elements.append(Paragraph("Learning Skills Sankey", styles["Heading3"]))
        elements.append(Image(sankey_path, width=7.0 * inch, height=4.1 * inch))
    except Exception:
        elements.append(Paragraph("Chart images could not be embedded into the PDF. Install kaleido to enable chart export: <b>pip install kaleido</b>", styles["BodyText"]))

    elements.append(Spacer(1, 0.15 * inch))
    elements.append(Paragraph("What I will do differently next time", styles["Heading3"]))
    table_df = student_reflection_table(student_df)
    data = [["Topic", "One thing I will do differently next time"]] + table_df.values.tolist()
    tbl = Table(data, repeatRows=1, colWidths=[1.8 * inch, 5.2 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    elements.append(tbl)
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


st.title("Learning Progress Dashboard")
st.caption("Upload an Excel or CSV file to analyse class patterns and export individual student reports.")

with st.sidebar:
    st.header("Upload data")
    uploaded = st.file_uploader("Upload Excel / CSV", type=["xlsx", "xls", "csv"])
    st.markdown(
        "**Expected fields**\n- Username\n- Marks for Current Topical Quiz\n- Preparation strategies\n- Learning skills\n- Performance issues\n- Topic\n- One thing I will do differently next time"
    )

if not uploaded:
    st.info("Upload your file to begin.")
    st.stop()

try:
    df = read_uploaded_file(uploaded)
except Exception as e:
    st.error(f"Could not process file: {e}")
    st.stop()

st.success(f"Loaded {len(df)} records for {df['Username'].nunique()} students.")

with st.expander("Preview cleaned data"):
    st.dataframe(df, use_container_width=True)

# Overview metrics
c1, c2, c3, c4 = st.columns(4)
c1.metric("Students", df["Username"].nunique())
c2.metric("Records", len(df))
c3.metric("Mean marks", f"{df['Marks'].mean():.1f}")
c4.metric("Median marks", f"{df['Marks'].median():.1f}")

tab1, tab2, tab3, tab4 = st.tabs(["Histogram", "Sankey", "Scatter", "Student Report"])

with tab1:
    st.subheader("Class Distribution of Scores")
    selected_topics = st.multiselect("Filter by topic", TOPIC_ORDER, default=TOPIC_ORDER)
    hist_df = df[df["Topic"].astype(str).isin(selected_topics)].copy()
    nbins = st.slider("Number of bins", 5, 30, 10)
    color_by_topic = st.toggle("Colour by topic", value=True)
    fig_hist = px.histogram(
        hist_df,
        x="Marks",
        color="Topic" if color_by_topic else None,
        nbins=nbins,
        marginal="box",
        barmode="overlay" if color_by_topic else "relative",
        title="Histogram of class scores",
        hover_data=["Username", "Topic"],
    )
    fig_hist.update_layout(height=520)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab2:
    st.subheader("Preparation Strategy Sankey by Topic")
    sankey_fig = build_topic_strategy_sankey(
        df[df["Topic"].astype(str).isin(TOPIC_ORDER)],
        "Preparation Strategies",
        "Preparation Strategies used across Topics",
    )
    st.plotly_chart(sankey_fig, use_container_width=True)
    st.caption("Links show how often each preparation strategy appeared within each topic.")

with tab3:
    st.subheader("Marks vs Reflection Categories")
    y_option = st.radio(
        "Choose the variable for the scatter plot",
        ["Learning Skills", "Performance Issues"],
        horizontal=True,
    )
    scatter_title = f"Marks vs {y_option}"
    scatter_fig = build_scatter(df, y_option, scatter_title)
    st.plotly_chart(scatter_fig, use_container_width=True)
    st.caption("Each point represents a student-category combination. Hover to see the username, topic and marks.")

with tab4:
    st.subheader("Individual Student Progress")
    students = sorted(df["Username"].dropna().unique().tolist())
    username = st.selectbox("Select a student", students)
    student_df = df[df["Username"] == username].copy().sort_values("Topic")

    left, right = st.columns([1.2, 1])
    with left:
        st.plotly_chart(build_student_line(student_df), use_container_width=True)
    with right:
        st.plotly_chart(build_student_skill_sankey(student_df), use_container_width=True)

    st.markdown("### Reflections")
    reflection_cols = [c for c in student_df.columns if c.startswith("Reflection ")]
    if reflection_cols:
        for _, row in student_df.iterrows():
            st.markdown(f"**{row['Topic']}**")
            for col in reflection_cols:
                if normalise_text(row[col]):
                    st.write(row[col])
    else:
        st.info("No narrative reflection columns detected.")

    st.markdown("### One thing I will do differently next time")
    st.dataframe(student_reflection_table(student_df), use_container_width=True, hide_index=True)

    pdf_bytes = build_student_pdf(student_df, username)
    st.download_button(
        label="Download individual PDF report",
        data=pdf_bytes,
        file_name=f"{username.replace('@', '_at_').replace('.', '_')}_learning_progress.pdf",
        mime="application/pdf",
    )
