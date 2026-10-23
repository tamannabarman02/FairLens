"""Plotly charts with a shared, theme-aware template. FairLens = accent; baselines = greys."""
from __future__ import annotations

import plotly.graph_objects as go

PALETTE = {
    "light": {"text": "#1B1A2E", "muted": "#6B6A80", "grid": "#ECEAF6", "accent": "#6D4AFF", "red": "#E5484D",
              "amber": "#D98A0B", "green": "#1F9D5C", "blue": "#2F6FEB", "grey": "#B9B7CC"},
    "dark": {"text": "#ECEBFA", "muted": "#A3A2BC", "grid": "#2E3050", "accent": "#8B70FF", "red": "#FF6369",
             "amber": "#FFB84D", "green": "#3DD68C", "blue": "#60A5FA", "grey": "#5C5E7A"},
}
SERIES = ["#6D4AFF", "#2F6FEB", "#1F9D5C", "#D98A0B", "#E5484D", "#9B8CFF"]
DETECTOR_COLORS = {"BERT": "#6D4AFF", "DeBERTa": "#2F6FEB", "BART-MNLI": "#1F9D5C", "Rule-based": "#D98A0B",
                   "Hub pretrained": "#9B8CFF", "Ensemble": "#E5484D"}
DECISION_COLORS = {"ACCEPT": "green", "REJECT": "red", "RETRY": "amber", "HUMAN_REVIEW": "blue"}


def style(fig: go.Figure, theme: str = "light", height: int = 340, legend: bool = True) -> go.Figure:
    p = PALETTE[theme]
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=13, color=p["text"]), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(font_family="Inter"),
    )
    fig.update_xaxes(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["grid"], tickfont=dict(color=p["muted"]))
    fig.update_yaxes(gridcolor=p["grid"], zerolinecolor=p["grid"], linecolor=p["grid"], tickfont=dict(color=p["muted"]))
    return fig


def bar(labels, values, theme="light", colors=None, errors=None, yrange=None, height=320, fmt=".2f", ytitle=None):
    fig = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors or [SERIES[i % len(SERIES)] for i in range(len(labels))],
        text=[format(v, fmt) if v is not None else "" for v in values], textposition="outside", cliponaxis=False,
        error_y=dict(type="data", symmetric=False, array=[e[1] for e in errors], arrayminus=[e[0] for e in errors],
                     color=PALETTE[theme]["muted"]) if errors else None,
        marker_line_width=0))
    fig.update_traces(marker_cornerradius=6)
    if yrange:
        fig.update_yaxes(range=yrange)
    if ytitle:
        fig.update_yaxes(title=ytitle)
    return style(fig, theme, height, legend=False)


def grouped_bar(categories, series: dict[str, list], theme="light", height=340, yrange=None, colors=None):
    fig = go.Figure()
    for i, (name, vals) in enumerate(series.items()):
        fig.add_bar(name=name, x=categories, y=vals, marker_color=(colors or {}).get(name, SERIES[i % len(SERIES)]),
                    marker_cornerradius=5)
    fig.update_layout(barmode="group")
    if yrange:
        fig.update_yaxes(range=yrange)
    return style(fig, theme, height)


def donut(labels, values, theme="light", colors=None, height=300):
    p = PALETTE[theme]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=.62, sort=False,
                           marker=dict(colors=[p[c] if c in p else c for c in (colors or SERIES)]),
                           textinfo="percent", textfont=dict(color="#fff")))
    return style(fig, theme, height)


def confusion(cm, labels=("Neutral", "Biased"), theme="light", height=260, title=None):
    p = PALETTE[theme]
    fig = go.Figure(go.Heatmap(z=cm, x=[f"Pred {l}" for l in labels], y=[f"True {l}" for l in labels],
                               colorscale=[[0, "rgba(109,74,255,0.06)"], [1, p["accent"]]], showscale=False,
                               text=cm, texttemplate="%{text}", textfont=dict(size=16)))
    fig.update_yaxes(autorange="reversed")
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14)))
    return style(fig, theme, height, legend=False)


def lines(series: dict[str, tuple[list, list]], theme="light", height=320, xtitle="", ytitle="", diagonal=False):
    fig = go.Figure()
    for i, (name, (x, y)) in enumerate(series.items()):
        fig.add_scatter(x=x, y=y, mode="lines", name=name, line=dict(width=2.5, color=DETECTOR_COLORS.get(name, SERIES[i % len(SERIES)])))
    if diagonal:
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfect calibration",
                        line=dict(dash="dot", color=PALETTE[theme]["muted"], width=1.5))
    fig.update_xaxes(title=xtitle, range=[0, 1])
    fig.update_yaxes(title=ytitle, range=[0, 1.02])
    return style(fig, theme, height)


def radar(names, values, theme="light", height=320, label="Candidate", compare: tuple[str, list] | None = None):
    p = PALETTE[theme]
    fig = go.Figure()
    fig.add_scatterpolar(r=values + values[:1], theta=names + names[:1], fill="toself", name=label,
                         line_color=p["accent"], fillcolor="rgba(109,74,255,0.18)")
    if compare:
        fig.add_scatterpolar(r=compare[1] + compare[1][:1], theta=names + names[:1], name=compare[0],
                             line=dict(color=p["grey"], dash="dot"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(range=[0, 1], gridcolor=p["grid"], tickfont=dict(color=p["muted"])),
                                 angularaxis=dict(gridcolor=p["grid"], tickfont=dict(color=p["text"]))))
    return style(fig, theme, height)


def hbar(labels, values, theme="light", colors=None, height=280, fmt=".2f", xrange=(0, 1)):
    fig = go.Figure(go.Bar(y=labels, x=values, orientation="h", marker_color=colors or SERIES[0],
                           text=[format(v, fmt) for v in values], textposition="outside", cliponaxis=False))
    fig.update_traces(marker_cornerradius=5)
    fig.update_xaxes(range=list(xrange))
    fig.update_yaxes(autorange="reversed")
    return style(fig, theme, height, legend=False)
