"""Professional, consistent Plotly charts for the dashboard."""
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler

LEVELS = ["Low", "Medium", "High", "Critical"]
RISK_COLORS = {"Low": "#94A3B8", "Medium": "#F2B01E", "High": "#E4572E", "Critical": "#9B1C31"}
MODEL_COLORS = {"IsolationForest": "#2563EB", "OneClassSVM": "#0D9488", "LOF": "#7C3AED",
                "Ensemble (>=2 votes)": "#111827"}
SHORT = {"IsolationForest": "IF", "OneClassSVM": "SVM", "LOF": "LOF"}
FONT = "Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
INK, MUTED, GRID = "#1F2937", "#6B7280", "#EEF2F7"
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def style(fig, title=None, subtitle=None, height=380, legend=True):
    text = f"<b>{title}</b>" if title else None
    if title and subtitle:
        text += f"<br><span style='font-size:12px;color:{MUTED}'>{subtitle}</span>"
    fig.update_layout(
        template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family=FONT, size=12.5, color=INK),
        title=dict(text=text, x=0.0, xanchor="left", y=0.97, font=dict(size=16)),
        margin=dict(l=12, r=12, t=78 if subtitle else 62 if title else 16, b=12),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11.5)),
        hoverlabel=dict(font_family=FONT, bgcolor="white", bordercolor="#CBD5E1"),
    )
    fig.update_xaxes(showline=True, linecolor="#CBD5E1", gridcolor=GRID, zeroline=False, ticks="outside",
                     tickcolor="#CBD5E1")
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


# ------------------------------------------------------------------ overview
def risk_donut(risk):
    counts = risk["risk_level"].value_counts().reindex(LEVELS).fillna(0).astype(int)
    fig = go.Figure(go.Pie(
        labels=counts.index, values=counts.values, hole=0.66, sort=False, direction="clockwise",
        marker=dict(colors=[RISK_COLORS[l] for l in counts.index], line=dict(color="white", width=3)),
        textinfo="value", textfont=dict(size=13, color="white"),
        hovertemplate="%{label}: %{value} users (%{percent})<extra></extra>"))
    fig.add_annotation(text=f"<b style='font-size:30px'>{len(risk)}</b><br>"
                            f"<span style='color:{MUTED}'>users</span>", showarrow=False)
    return style(fig, "Risk distribution", "Users by number of models that flagged them", 380)


def user_map(feats, risk, feature_cols):
    """2-D PCA projection of users; colour = risk level."""
    X = StandardScaler().fit_transform(feats[feature_cols])
    pca = PCA(n_components=2, random_state=42).fit(X)
    xy = pca.transform(X)
    df = feats[["user"]].copy()
    df["x"], df["y"] = xy[:, 0], xy[:, 1]
    df = df.merge(risk[["user", "risk_level", "anomaly_votes"]], on="user")
    var = pca.explained_variance_ratio_ * 100
    fig = go.Figure()
    for lvl in LEVELS:
        d = df[df["risk_level"] == lvl]
        fig.add_trace(go.Scatter(
            x=d["x"], y=d["y"], mode="markers", name=lvl, text=d["user"],
            marker=dict(size=7 if lvl == "Low" else 13, color=RISK_COLORS[lvl],
                        opacity=0.55 if lvl == "Low" else 0.95,
                        line=dict(width=1.2 if lvl != "Low" else 0, color="white")),
            hovertemplate="<b>%{text}</b><br>" + lvl + " risk<extra></extra>"))
    fig.update_xaxes(title=f"Principal component 1 ({var[0]:.0f}% of variance)")
    fig.update_yaxes(title=f"Principal component 2 ({var[1]:.0f}% of variance)")
    return style(fig, "Behaviour map of users", "Each dot is a user; distant dots behave differently", 380)


def activity_heatmap(clean):
    ts = clean["timestamp"]
    z = (pd.crosstab(ts.dt.dayofweek, ts.dt.hour).reindex(index=range(7), columns=range(24), fill_value=0))
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[f"{h:02d}" for h in range(24)], y=DAYS, colorscale="Blues", xgap=2, ygap=2,
        colorbar=dict(title="Events", thickness=12, len=0.8),
        hovertemplate="%{y} %{x}:00<br>%{z:,} events<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(title="Hour of day", showgrid=False)
    return style(fig, "Logon activity by weekday and hour", "Total events across all users", 340, legend=False)


def hourly_profile(clean):
    d = clean.assign(hour=clean["timestamp"].dt.hour,
                     result=np.where(clean["event_id"] == 4625, "Failed", "Successful"))
    t = d.groupby(["hour", "result"]).size().unstack(fill_value=0).reindex(range(24), fill_value=0)
    fig = go.Figure()
    for col, color in [("Successful", "#2563EB"), ("Failed", "#E4572E")]:
        if col in t:
            fig.add_bar(x=t.index, y=t[col], name=col, marker_color=color)
    fig.update_layout(barmode="stack", bargap=0.2)
    fig.update_xaxes(title="Hour of day", dtick=2)
    fig.update_yaxes(title="Events")
    return style(fig, "Logons by hour", "Successful vs failed", 340)


# ---------------------------------------------------------------- evaluation
def metrics_bars(summary):
    fig = go.Figure()
    for metric, color in [("Precision", "#2563EB"), ("Recall", "#0D9488"), ("F1", "#7C3AED")]:
        fig.add_bar(x=summary["Model"], y=summary[metric], name=metric, marker_color=color,
                    text=summary[metric].map("{:.2f}".format), textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
    fig.update_yaxes(range=[0, 1.12], title="Score")
    return style(fig, "Precision, recall and F1 by model", None, 380)


def confusion_grid(summary):
    n = len(summary)
    fig = make_subplots(rows=1, cols=n, subplot_titles=list(summary["Model"]), horizontal_spacing=0.06)
    for i, (_, r) in enumerate(summary.iterrows(), start=1):
        cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
        fig.add_trace(go.Heatmap(
            z=cm / cm.sum(axis=1, keepdims=True).clip(min=1), zmin=0, zmax=1, x=["Normal", "Anomaly"],
            y=["Normal", "Anomaly"], text=cm, texttemplate="<b>%{text}</b>", textfont=dict(size=17),
            customdata=cm, colorscale=[[0, "#F1F5F9"], [1, "#2563EB"]], showscale=False,
            xgap=3, ygap=3, hovertemplate="Actual %{y}<br>Predicted %{x}<br>%{customdata} users<extra></extra>"),
            row=1, col=i)
        fig.update_yaxes(autorange="reversed", row=1, col=i, showgrid=False,
                         title="Actual" if i == 1 else None)
        fig.update_xaxes(row=1, col=i, showgrid=False, title="Predicted")
    fig.update_annotations(font=dict(size=13, family=FONT))
    return style(fig, "Confusion matrices", "Counts of users; shading shows the share of each actual class", 360, legend=False)


def roc_pr(results, truth):
    """ROC and precision-recall curves from the models' continuous anomaly scores."""
    y = truth.to_numpy()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ROC curve", "Precision-recall curve"),
                        horizontal_spacing=0.1)
    for name, col in [("IsolationForest", "IF_score"), ("OneClassSVM", "SVM_score"), ("LOF", "LOF_score")]:
        s = -results[col].to_numpy()                       # higher = more anomalous
        fpr, tpr, _ = roc_curve(y, s)
        p, r, _ = precision_recall_curve(y, s)
        c = MODEL_COLORS[name]
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=c, width=2.5),
                                 name=f"{name} (AUC {auc(fpr, tpr):.2f})", legendgroup=name), row=1, col=1)
        fig.add_trace(go.Scatter(x=r, y=p, mode="lines", line=dict(color=c, width=2.5),
                                 name=f"{name} (AP {average_precision_score(y, s):.2f})",
                                 legendgroup=name, showlegend=False,
                                 hovertemplate=f"{name}: AP {average_precision_score(y, s):.2f}<extra></extra>"),
                      row=1, col=2)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#9CA3AF", dash="dash"),
                             showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_hline(y=y.mean(), line=dict(color="#9CA3AF", dash="dash"), row=1, col=2)
    fig.update_xaxes(title="False positive rate", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="True positive rate", range=[0, 1.02], row=1, col=1)
    fig.update_xaxes(title="Recall", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title="Precision", range=[0, 1.02], row=1, col=2)
    fig.update_annotations(font=dict(size=13, family=FONT))
    style(fig, "Ranking quality of the raw anomaly scores", "Independent of the 5% flagging threshold", 440)
    fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.2, x=0.5, xanchor="center"),
                      margin=dict(b=70))
    return fig


def kappa_heatmap(kappa):
    z = kappa.to_numpy()
    fig = go.Figure(go.Heatmap(
        z=z, x=list(kappa.columns), y=list(kappa.index), zmin=0, zmax=1, xgap=3, ygap=3,
        text=np.round(z, 2), texttemplate="<b>%{text}</b>", textfont=dict(size=15),
        colorscale=[[0, "#F1F5F9"], [1, "#0F766E"]], colorbar=dict(title="κ", thickness=12)))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False)
    return style(fig, "Model consistency", "Cohen's kappa between model decisions (1 = identical)", 360,
                 legend=False)


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
    colors = {1: RISK_COLORS["Medium"], 2: RISK_COLORS["High"], 3: RISK_COLORS["Critical"]}
    fig = go.Figure(go.Bar(x=d["users"], y=d["combo"], orientation="h", marker_color=d["k"].map(colors),
                           text=d["users"], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="Users")
    fig.update_yaxes(showgrid=False)
    return style(fig, "Which models flag the same users", "Exact combinations of detectors", 360, legend=False)


# --------------------------------------------------------------------- users
def zscores(feats, feature_cols):
    num = feats.set_index("user")[feature_cols]
    return (num - num.mean()) / num.std().replace(0, 1)


def feature_heatmap(z_flagged):
    z = z_flagged.clip(-4, 4)
    fig = go.Figure(go.Heatmap(
        z=z.values, x=[c.replace("_", " ") for c in z.columns], y=z.index, zmid=0, zmin=-4, zmax=4,
        colorscale="RdBu_r", xgap=1, ygap=1, colorbar=dict(title="z", thickness=12),
        hovertemplate="%{y}<br>%{x}: z=%{z:.1f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, tickangle=-40)
    return style(fig, "What makes flagged users different", "Feature z-scores vs. all users (red = higher)",
                 max(320, 28 * len(z) + 190), legend=False)


def zscore_bar(z_user, top=10):
    s = z_user.reindex(z_user.abs().sort_values(ascending=False).index)[:top][::-1]
    fig = go.Figure(go.Bar(
        x=s.values, y=[i.replace("_", " ") for i in s.index], orientation="h",
        marker_color=["#E4572E" if v > 0 else "#2563EB" for v in s.values],
        text=[f"{v:+.1f}" for v in s.values], textposition="outside", cliponaxis=False))
    fig.update_xaxes(title="Standard deviations from the population mean")
    fig.update_yaxes(showgrid=False)
    return style(fig, "Top behavioural deviations", None, 380, legend=False)


def user_timeline(ev):
    ev = ev.assign(hour=ev["timestamp"].dt.hour + ev["timestamp"].dt.minute / 60,
                   result=np.where(ev["event_id"] == 4625, "Failed", "Successful"))
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=7, fillcolor="#E2E8F0", opacity=0.55, line_width=0)
    fig.add_hrect(y0=19, y1=24, fillcolor="#E2E8F0", opacity=0.55, line_width=0)
    for res, color, size in [("Successful", "#2563EB", 6), ("Failed", "#E4572E", 8)]:
        d = ev[ev["result"] == res]
        fig.add_trace(go.Scattergl(x=d["timestamp"], y=d["hour"], mode="markers", name=res,
                                   marker=dict(color=color, size=size, opacity=0.8),
                                   text=d["pc"], hovertemplate="%{x}<br>PC/IP: %{text}<extra></extra>"))
    fig.update_yaxes(title="Hour of day", range=[0, 24], dtick=4)
    return style(fig, "Event timeline", "Shaded bands are off-hours (before 07:00, after 19:00)", 340)


# -------------------------------------------------------------------- alerts
def alerts_timeline(alerts):
    a = alerts.assign(time=pd.to_datetime(alerts["time"]))
    fig = go.Figure()
    for lvl in ["Medium", "High", "Critical"]:
        d = a[a["risk_level"] == lvl]
        if len(d):
            fig.add_trace(go.Scatter(x=d["time"], y=d["user"], mode="markers", name=lvl,
                                     marker=dict(size=13, color=RISK_COLORS[lvl],
                                                 line=dict(width=1.5, color="white")),
                                     hovertemplate="<b>%{y}</b><br>%{x}<extra>" + lvl + "</extra>"))
    fig.update_yaxes(type="category", showgrid=True)
    return style(fig, "Alert timeline", "When each user crossed the alert threshold",
                 max(300, 26 * a["user"].nunique() + 150))
