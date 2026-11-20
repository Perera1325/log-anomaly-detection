from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

LEVELS = ["Low", "Medium", "High", "Critical"]
# Okabe-Ito colour-blind-safe palette (Okabe & Ito, 2008)
BLUE, ORANGE, GREEN, VERMILION, PURPLE, SKY, BLACK = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7",
                                                      "#56B4E9", "#000000")
RISK_COLORS = {"Low": "#9A9A9A", "Medium": ORANGE, "High": VERMILION, "Critical": "#7F1D1D"}
MODEL_COLORS = {"IsolationForest": BLUE, "OneClassSVM": GREEN, "LOF": PURPLE, "Ensemble (>=2 votes)": BLACK}
MODEL_DASH = {"IsolationForest": "solid", "OneClassSVM": "dash", "LOF": "dot"}
SHORT = {"IsolationForest": "IF", "OneClassSVM": "SVM", "LOF": "LOF"}
FONT = 'Georgia, "Times New Roman", Times, serif'
INK, GRID, AXIS = "#111111", "#E6E6E6", "#222222"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
SEQ = [[0, "#FFFFFF"], [1, "#08306B"]]
DIV = [[0, BLUE], [0.5, "#FFFFFF"], [1, VERMILION]]


def style(fig, title=None, subtitle=None, height=360, legend=True):
    """title/subtitle are accepted for compatibility but not drawn (captions live under the figure)."""
    fig.update_layout(
        template="simple_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT, size=13, color=INK), margin=dict(l=10, r=10, t=34, b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12),
                    bgcolor="rgba(255,255,255,0)"),
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor="#999"))
    fig.update_xaxes(showline=True, mirror=True, linecolor=AXIS, linewidth=1, ticks="outside", tickcolor=AXIS,
                     gridcolor=GRID, zeroline=False, title_font=dict(size=13))
    fig.update_yaxes(showline=True, mirror=True, linecolor=AXIS, linewidth=1, ticks="outside", tickcolor=AXIS,
                     gridcolor=GRID, zeroline=False, title_font=dict(size=13))
    return fig


def _flat(fig):
    """Heatmaps / categorical bars: no grid."""
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


# ------------------------------------------------------------------ overview
def risk_donut(risk):
    """Users per risk level (horizontal bars; the name is kept for backwards compatibility)."""
    counts = risk["risk_level"].value_counts().reindex(LEVELS).fillna(0).astype(int)
    share = counts / counts.sum() * 100
    fig = go.Figure(go.Bar(
        x=counts.values, y=counts.index, orientation="h", marker=dict(color=[RISK_COLORS[l] for l in counts.index],
                                                                      line=dict(color=AXIS, width=0.8)),
        text=[f"{n:,}  ({p:.1f}%)" for n, p in zip(counts.values, share)], textposition="outside",
        cliponaxis=False, hovertemplate="%{y}: %{x:,} users<extra></extra>"))
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_xaxes(title="Number of accounts", range=[0, counts.max() * 1.5])
    return _flat(style(fig, height=300, legend=False))


def user_map(feats, risk, feature_cols):
    """2-D PCA projection of users; marker shape and colour encode the risk level."""
    X = StandardScaler().fit_transform(feats[feature_cols])
    pca = PCA(n_components=2, random_state=42).fit(X)
    xy = pca.transform(X)
    df = feats[["user"]].copy()
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    df = df.merge(risk[["user", "risk_level", "anomaly_votes"]], on="user")
    var = pca.explained_variance_ratio_ * 100
    symbols = {"Low": "circle", "Medium": "diamond", "High": "square", "Critical": "x"}
    fig = go.Figure()
    for lvl in LEVELS:
        d = df[df["risk_level"] == lvl]
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"], mode="markers", name=lvl, text=d["user"],
            marker=dict(size=6 if lvl == "Low" else 10, symbol=symbols[lvl], color=RISK_COLORS[lvl],
                        opacity=0.5 if lvl == "Low" else 0.95,
                        line=dict(width=0.8 if lvl not in ("Low", "Critical") else 0, color=AXIS)),
            hovertemplate="<b>%{text}</b><br>" + lvl + " risk<extra></extra>"))
    fig.update_xaxes(title=f"Principal component 1 ({var[0]:.0f}% of variance)")
    fig.update_yaxes(title=f"Principal component 2 ({var[1]:.0f}% of variance)")
    return style(fig, height=360)


def user_map_3d(feats, risk, feature_cols):
    """3-D PCA projection of users (interactive; the dashboard rotates it slowly and fans the points out on first view)."""
    X = StandardScaler().fit_transform(feats[feature_cols])
    pca = PCA(n_components=3, random_state=42).fit(X)
    p = pca.transform(X)
    df = feats[["user"]].copy()
    df["x"], df["y"], df["z"] = p[:, 0], p[:, 1], p[:, 2]
    df = df.merge(risk[["user", "risk_level"]], on="user")
    var = pca.explained_variance_ratio_ * 100
    sym = {"Low": "circle", "Medium": "diamond", "High": "square", "Critical": "x"}
    size = {"Low": 2.5, "Medium": 4, "High": 6, "Critical": 8}
    fig = go.Figure()
    for votes, lvl in enumerate(LEVELS):
        d = df[df["risk_level"] == lvl]
        fig.add_trace(go.Scatter3d(
            x=d["x"], y=d["y"], z=d["z"], mode="markers", name=lvl, text=d["user"],
            marker=dict(size=size[lvl], symbol=sym[lvl], color=RISK_COLORS[lvl],
                        opacity=0.45 if lvl == "Low" else 0.95,
                        line=dict(width=0.6 if lvl in ("Medium", "High") else 0, color=AXIS)),
            hovertemplate="<b>%{text}</b><br>" + f"{lvl} risk ({votes} of 3 detectors)<extra></extra>"))
    axis = dict(showbackground=True, backgroundcolor="#FAFAFA", gridcolor=GRID, zerolinecolor=GRID,
                linecolor=AXIS, showline=True, ticks="", showticklabels=False, title_font=dict(size=12))
    fig.update_layout(scene=dict(
        xaxis=dict(title=f"PC1 ({var[0]:.0f}%)", **axis), yaxis=dict(title=f"PC2 ({var[1]:.0f}%)", **axis),
        zaxis=dict(title=f"PC3 ({var[2]:.0f}%)", **axis), aspectmode="cube",
        camera=dict(eye=dict(x=2.0, y=1.0, z=0.85))))
    style(fig, height=520)
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0),
                      legend=dict(x=0, xanchor="left", y=1.0, yanchor="top", orientation="v"))
    return fig


def activity_heatmap(clean):
    ts = clean["timestamp"]
    z = pd.crosstab(ts.dt.dayofweek, ts.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[f"{h:02d}" for h in range(24)], y=DAYS, colorscale=SEQ, xgap=1, ygap=1,
        colorbar=dict(title="Events", thickness=12, len=0.9, outlinewidth=0.8),
        hovertemplate="%{y} %{x}:00<br>%{z:,} events<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Hour of day")
    return _flat(style(fig, height=330, legend=False))


def hourly_profile(clean):
    d = clean.assign(hour=clean["timestamp"].dt.hour,
                     result=np.where(clean["event_id"] == 4625, "Failed", "Successful"))
    t = d.groupby(["hour", "result"]).size().unstack(fill_value=0).reindex(range(24), fill_value=0)
    fig = go.Figure()
    for col, color in [("Successful", BLUE), ("Failed", VERMILION)]:
        if col in t:
            fig.add_bar(x=t.index, y=t[col], name=col, marker=dict(color=color, line=dict(color=AXIS, width=0.5)))
    fig.update_layout(barmode="stack", bargap=0.15)
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_yaxes(title="Events")
    return style(fig, height=330)


# ---------------------------------------------------------------- evaluation
def metrics_bars(summary):
    fig = go.Figure()
    for metric, color in [("Precision", BLUE), ("Recall", ORANGE), ("F1", "#555555")]:
        fig.add_bar(x=summary["Model"], y=summary[metric], name=metric,
                    marker=dict(color=color, line=dict(color=AXIS, width=0.6)),
                    text=summary[metric].map("{:.2f}".format), textposition="outside", cliponaxis=False,
                    textfont=dict(size=10.5))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.05)
    fig.update_yaxes(range=[0, 1.1], title="Score")
    fig.update_xaxes(showgrid=False)
    return style(fig, height=360)


def confusion_grid(summary):
    n = len(summary)
    letters = "abcdefgh"
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=[f"({letters[i]}) {m}" for i, m in enumerate(summary["Model"])])
    for i, (_, r) in enumerate(summary.iterrows(), start=1):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        fig.add_trace(go.Heatmap(
            z=cm / cm.sum(axis=1, keepdims=True).clip(min=1), zmin=0, zmax=1, x=["Normal", "Anomaly"],
            y=["Normal", "Anomaly"], text=cm, texttemplate="%{text}", textfont=dict(size=15), customdata=cm,
            colorscale=SEQ, showscale=False, xgap=2, ygap=2,
            hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{customdata} users<extra></extra>"), row=1, col=i)
        fig.update_xaxes(row=1, col=i, showgrid=False, title="Predicted" if i == (n + 1) // 2 else None)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_yaxes(title="Actual", row=1, col=1)
    fig.update_annotations(font=dict(size=12.5, family=FONT))
    return style(fig, height=330, legend=False)


def roc_pr(results, truth):
    """ROC and precision-recall curves from the detectors' continuous anomaly scores."""
    y = truth.to_numpy()
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=("(a) Receiver operating characteristic", "(b) Precision-recall"))
    for name, col in [("IsolationForest", "IF_score"), ("OneClassSVM", "SVM_score"), ("LOF", "LOF_score")]:
        s = -results[col].to_numpy()                       # higher = more anomalous
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        line = dict(color=MODEL_COLORS[name], width=2.2, dash=MODEL_DASH[name])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=line, legendgroup=name,
                                 name=f"{name} (AUC = {auc(fpr, tpr):.3f})"), row=1, col=1)
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", line=line, legendgroup=name, showlegend=False,
                                 hovertemplate=f"{name}: AP = {average_precision_score(y, s):.3f}<extra></extra>"),
                      row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#888", dash="dashdot", width=1),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_hline(y=y.mean(), line=dict(color="#888", dash="dashdot", width=1), row=1, col=2)
    fig.update_xaxes(title="False positive rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="True positive rate", range=[0, 1.02], row=1, col=1)
    fig.update_xaxes(title="Recall", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title="Precision", range=[0, 1.02], row=1, col=2)
    fig.update_annotations(font=dict(size=12.5, family=FONT))
    style(fig, height=420)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, x=0.5, xanchor="center"),
                      margin=dict(b=70))
    return fig


def kappa_heatmap(kappa):
    z = kappa.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=list(kappa.columns), y=list(kappa.index), zmin=0, zmax=1, xgap=2, ygap=2, text=np.round(z, 2),
        texttemplate="%{text}", textfont=dict(size=14), colorscale=[[0, "#FFFFFF"], [1, "#0B5D52"]],
        colorbar=dict(title="kappa", thickness=12, outlinewidth=0.8)))
    fig.update_yaxes(autorange="reversed")
    return _flat(style(fig, height=330, legend=False))


def vote_combinations(results):
    models = ["IsolationForest", "OneClassSVM", "LOF"]
    flags = pd.DataFrame({m: results[m] == -1 for m in models})
    rows = []
    for k in (1, 2, 3):
        for combo in combinations(models, k):
            others = [m for m in models if m not in combo]
            mask = flags[list(combo)].all(axis=1) & ~flags[others].any(axis=1) if others else flags.all(axis=1)
            rows.append(("+".join(SHORT[m] for m in combo), int(mask.sum()), k))
    d = pd.DataFrame(rows, columns=["combo", "users", "k"]).sort_values(["k", "users"])
    colors = {1: ORANGE, 2: VERMILION, 3: RISK_COLORS["Critical"]}
    fig = go.Figure(go.Bar(x=d["users"], y=d["combo"], orientation="h",
                           marker=dict(color=d["k"].map(colors), line=dict(color=AXIS, width=0.6)),
                           text=d["users"], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="Number of accounts", range=[0, max(d["users"].max() * 1.15, 1)])
    return _flat(style(fig, height=330, legend=False))


# --------------------------------------------------------------------- users
def zscores(feats, feature_cols):
    num = feats.set_index("user")[feature_cols]
    return (num - num.mean()) / num.std().replace(0, 1)


def feature_heatmap(z_flagged):
    z = z_flagged.clip(-4, 4)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[c.replace("_", " ") for c in z.columns], y=z.index, zmid=0, zmin=-4, zmax=4,
        colorscale=DIV, xgap=1, ygap=1, colorbar=dict(title="z-score", thickness=12, outlinewidth=0.8),
        hovertemplate="%{y}<br>%{x}: z = %{z:.1f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-45)
    return _flat(style(fig, height=max(320, 26 * len(z) + 200), legend=False))


def zscore_bar(z_user, top=10):
    s = z_user.reindex(z_user.abs().sort_values(ascending=False).index)[:top][::-1]
    fig = go.Figure(go.Bar(
        x=s.values, y=[i.replace("_", " ") for i in s.index], orientation="h",
        marker=dict(color=[VERMILION if v > 0 else BLUE for v in s.values], line=dict(color=AXIS, width=0.6)),
        text=[f"{v:+.1f}" for v in s.values], textposition="outside", cliponaxis=False))
    lo, hi = min(s.min(), 0), max(s.max(), 0)
    pad = (hi - lo) * 0.14 or 1
    fig.update_xaxes(title="Deviation from the population mean (standard deviations)",
                     range=[lo - (pad if lo < 0 else 0), hi + pad])
    fig.update_yaxes(showgrid=False)
    return style(fig, height=360, legend=False)


def user_timeline(ev):
    ev = ev.assign(hour=ev["timestamp"].dt.hour + ev["timestamp"].dt.minute / 60,
                   result=np.where(ev["event_id"] == 4625, "Failed", "Successful"))
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=7, fillcolor="#EDEDED", opacity=0.9, line_width=0, layer="below")
    fig.add_hrect(y0=19, y1=24, fillcolor="#EDEDED", opacity=0.9, line_width=0, layer="below")
    for res, color, symbol, size in [("Successful", BLUE, "circle", 6), ("Failed", VERMILION, "x", 8)]:
        d = ev[ev["result"] == res]
        fig.add_trace(go.Scattergl(x=d["timestamp"], y=d["hour"], mode="markers", name=res,
                                   marker=dict(color=color, size=size, symbol=symbol, opacity=0.8),
                                   text=d["pc"], hovertemplate="%{x}<br>Source: %{text}<extra></extra>"))
    fig.update_yaxes(title="Hour of day", range=[0, 24], dtick=4)
    fig.update_xaxes(title="Time")
    return style(fig, height=330)


# -------------------------------------------------------------------- alerts
def alerts_timeline(alerts):
    a = alerts.assign(time=pd.to_datetime(alerts["time"]))
    symbols = {"Medium": "diamond", "High": "square", "Critical": "x"}
    fig = go.Figure()
    for lvl in ["Medium", "High", "Critical"]:
        d = a[a["risk_level"] == lvl]
        if len(d):
            fig.add_trace(go.Scatter(x=d["time"], y=d["user"], mode="markers", name=lvl,
                                     marker=dict(size=10, color=RISK_COLORS[lvl], symbol=symbols[lvl],
                                                 line=dict(width=0.8, color=AXIS) if lvl != "Critical" else None),
                                     hovertemplate="<b>%{y}</b><br>%{x}<extra>" + lvl + "</extra>"))
    fig.update_yaxes(type="category")
    fig.update_xaxes(title="Time")
    return style(fig, height=max(300, 24 * a["user"].nunique() + 140))


# ------------------------------------------------------- uncertainty / what-if
def ci_plot(ci):
    """Dot-and-whisker plot of point estimates with 95% bootstrap intervals, one panel per metric."""
    metrics = ["Precision", "Recall", "F1", "ROC AUC", "Avg. precision"]
    models = [m for m in ["IsolationForest", "OneClassSVM", "LOF", "Ensemble (>=2 votes)"] if m in set(ci["Model"])]
    letters = "abcde"
    fig = make_subplots(rows=1, cols=len(metrics), shared_yaxes=True, horizontal_spacing=0.03,
                        subplot_titles=[f"({letters[i]}) {m}" for i, m in enumerate(metrics)])
    for j, metric in enumerate(metrics, start=1):
        d = ci[ci["Metric"] == metric].set_index("Model").reindex(models).dropna(subset=["Estimate"])
        fig.add_trace(go.Scatter(
            x=d["Estimate"], y=d.index, mode="markers", showlegend=False,
            marker=dict(size=9, color=[MODEL_COLORS[m] for m in d.index], line=dict(width=0.8, color=AXIS)),
            error_x=dict(type="data", symmetric=False, array=(d["Upper"] - d["Estimate"]).to_numpy(),
                         arrayminus=(d["Estimate"] - d["Lower"]).to_numpy(), thickness=1.6, width=5, color=AXIS),
            customdata=np.c_[d["Lower"], d["Upper"]],
            hovertemplate="%{y}<br>%{x:.3f} (95% CI %{customdata[0]:.3f} to %{customdata[1]:.3f})<extra></extra>"),
            row=1, col=j)
        fig.update_xaxes(range=[0, 1], dtick=0.25, row=1, col=j, showgrid=True)
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=models)
    fig.update_annotations(font=dict(size=12, family=FONT))
    return style(fig, height=300, legend=False)


def operating_chart(curve, rate):
    """Precision, recall and F1 as the flagged share of accounts grows; the chosen operating point is marked."""
    fig = go.Figure()
    for metric, color, dash in [("Precision", BLUE, "solid"), ("Recall", ORANGE, "dash"), ("F1", BLACK, "dot")]:
        fig.add_trace(go.Scatter(x=curve["rate"] * 100, y=curve[metric], mode="lines", name=metric,
                                 line=dict(color=color, width=2.2, dash=dash)))
    fig.add_vline(x=rate * 100, line=dict(color="#888", dash="dashdot", width=1))
    fig.update_xaxes(title="Share of accounts flagged (%)")
    fig.update_yaxes(title="Score", range=[0, 1.02])
    return style(fig, height=330)
