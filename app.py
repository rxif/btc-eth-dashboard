import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, Input, Output, State, dash_table

from dashboard.terminal_theme import C, empty_figure, TABLE_STYLE

from pricing.svi import surface_iv, svi, butterfly_min_on_range
from pricing.metrics import smile_table, GEX
from dashboard.data import list_snapshots, build, k_range_at, compute_vrp, load_raw

NO_DATA = "No usable data for this snapshot"

app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY], title="BTC & ETH Dashboard")
server = app.server


def kpi(label, id_):
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div("-", id=id_, className="kpi-value"),
    ], className="glass kpi")


def snapshot_parts(f):
    """Split a snapshot path into (year, month, day, "HH:MM")."""
    year, month, day = f.parent.name[5:].split("-")
    return year, month, day, f"{f.stem[9:11]}:{f.stem[11:13]}"


def snapshots(ccy, year=None, month=None, day=None):
    """Snapshots of a currency, optionally filtered by year / month / day."""
    out = []
    for f in list_snapshots(ccy):
        try:
            y, m, d, t = snapshot_parts(f)
        except ValueError:
            continue
        if (year is None or y == year) and (month is None or m == month) \
                and (day is None or d == day):
            out.append((y, m, d, t, f))
    return out


def keep_or_latest(values, current):
    """Keep the current choice if it still exists, otherwise take the latest one."""
    if not values:
        return None
    return current if current in values else values[-1]

def skew_table(table):
    t = table.copy()
    for c in ("atm", "rr25", "bf25"):
        t[c] = (t[c] * 100).round(2)
    t["days"] = t["days"].round(1)
    t["skew"] = t["skew"].round(3)
    t["in_range"] = np.where(t["in_range"], "yes", "no")

    return dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in t.columns],
        data=t.to_dict("records"),
        style_data_conditional=[
            {"if": {"filter_query": '{in_range} = "no"'}, "color": C.MUTED},
        ],
        **TABLE_STYLE,
    )


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1(id="title", className="brand mt-4 mb-0"),
            html.P("Deribit options  |  Black-76  |  SVI surface", className="subtitle"),
        ], md=12),
    ]),
    dbc.Row([
        dbc.Col([
            dbc.Label("Currency"),
            dbc.Select(id="ccy", value="BTC", options=["BTC", "ETH"]),
        ], md=2),
        dbc.Col([dbc.Label("Year"), dbc.Select(id="snap-year")], md=2),
        dbc.Col([dbc.Label("Month"), dbc.Select(id="snap-month")], md=2),
        dbc.Col([dbc.Label("Day"), dbc.Select(id="snap-day")], md=2),
        dbc.Col([dbc.Label("Time (UTC)"), dbc.Select(id="snap")], md=2),
    ], className="mb-3 g-3 align-items-end"),

    dbc.Tabs([
        dbc.Tab(label="Smile", tab_id="smile", children=[
            dbc.Row([
                dbc.Col([dbc.Label("Expiry"), dbc.Select(id="expiry")], md=3),
            ], className="my-3"),
            dbc.Row([
                dbc.Col(kpi("Quoted points", "k-n"), md=True),
                dbc.Col(kpi("Forward (USD)", "k-f"), md=True),
                dbc.Col(kpi("ATM vol", "k-atm"), md=True),
                dbc.Col(kpi("Fit RMSE (vol pts)", "k-rmse"), md=True),
                dbc.Col(kpi("Butterfly g min", "k-g"), md=True),
            ], className="mb-3 g-3"),
            html.Div(dcc.Graph(id="smile-graph", config={"displaylogo": False}),
                     className="glass p-2"),
        ]),
        dbc.Tab(label="Surface", tab_id="surface", children=[
            html.Div(dcc.Graph(id="surface-graph", config={"displaylogo": False}),
                     className="glass p-2 mt-3"),
            html.P("Surface shown only where options are quoted. No extrapolation.",
                   className="footer-note mt-2"),
        ]),
        dbc.Tab(label="Skew", tab_id="skew", children=[
            dbc.Row([
                dbc.Col(html.Div(id="skew-table", className="glass p-2"), md=12),
            ], className="my-3"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="rrbf-graph", config={"displaylogo": False}),
                         className="glass p-2"), md=6),
                dbc.Col(html.Div(dcc.Graph(id="atm-graph", config={"displaylogo": False}),
                         className="glass p-2"), md=6),
            ], className="mb-3 g-3"),
            html.P("Greyed points: 25-delta strikes fall outside the quoted range (extrapolated).",
                className="footer-note mt-2"),
        ]),
        dbc.Tab(label="GEX", tab_id="gex", children=[
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="gex-graph", config={"displaylogo": False}),
                         className="glass p-2"), md=12),
            ], className="mb-3 g-3"),
        ]),
        dbc.Tab(label="VRP", tab_id="vrp", children = [
            dbc.Row([
                dbc.Col(kpi("IV30", "iv30"), md=True),
                dbc.Col(kpi("RV close to close", "rv-ctc"), md=True),
                dbc.Col(kpi("RV parkinson", "rv-p"), md=True),
                dbc.Col(kpi("VRP", "vrp"), md=True),
            ], className="mb-3 g-3"),
            dbc.Row([
                dbc.Col(html.Div(dcc.Graph(id="vrp-graph", config={"displaylogo": False}),
                         className="glass p-2"), md=12),
            ], className="mb-3 g-3")
        ])

    ], active_tab="smile"),

], fluid=False, className="pb-5")


@app.callback(Output("title", "children"), Input("ccy", "value"))
def set_title(ccy):
    return f"{ccy} Implied Volatility"


@app.callback(
    Output("snap-year", "options"), Output("snap-year", "value"),
    Input("ccy", "value"), State("snap-year", "value"),
)
def fill_years(ccy, current):
    years = sorted({y for y, *_ in snapshots(ccy)})
    return years, keep_or_latest(years, current)


@app.callback(
    Output("snap-month", "options"), Output("snap-month", "value"),
    Input("ccy", "value"), Input("snap-year", "value"), State("snap-month", "value"),
)
def fill_months(ccy, year, current):
    if not year:
        return [], None
    months = sorted({m for _, m, *_ in snapshots(ccy, year)})
    opts = [{"label": calendar.month_name[int(m)], "value": m} for m in months]
    return opts, keep_or_latest(months, current)


@app.callback(
    Output("snap-day", "options"), Output("snap-day", "value"),
    Input("ccy", "value"), Input("snap-year", "value"), Input("snap-month", "value"),
    State("snap-day", "value"),
)
def fill_days(ccy, year, month, current):
    if not (year and month):
        return [], None
    days = sorted({d for _, _, d, *_ in snapshots(ccy, year, month)})
    opts = [{"label": str(int(d)), "value": d} for d in days]
    return opts, keep_or_latest(days, current)


@app.callback(
    Output("snap", "options"), Output("snap", "value"),
    Input("ccy", "value"), Input("snap-year", "value"), Input("snap-month", "value"),
    Input("snap-day", "value"), State("snap", "value"),
)
def fill_times(ccy, year, month, day, current):
    if not (year and month and day):
        return [], None
    items = sorted(snapshots(ccy, year, month, day), key=lambda s: s[3])
    opts = [{"label": t, "value": str(f)} for *_, t, f in items]
    # Keep the same time of day when switching day, if it exists
    current_time = snapshot_parts(Path(current))[3] if current else None
    paths_by_time = {t: str(f) for *_, t, f in items}
    times = list(paths_by_time)
    chosen = keep_or_latest(times, current_time)
    return opts, (paths_by_time[chosen] if chosen else None)


@app.callback(Output("expiry", "options"), Output("expiry", "value"), Input("snap", "value"))
def fill_expiries(path):
    if not path:
        return [], None
    _, slices, _ = build(path)
    opts = [{"label": f"{T * 365:.1f} days", "value": str(i)} for i, (T, _) in enumerate(slices)]
    return opts, ("0" if slices else None)


@app.callback(
    Output("smile-graph", "figure"), Output("k-n", "children"), Output("k-f", "children"),
    Output("k-atm", "children"), Output("k-rmse", "children"), Output("k-g", "children"),
    Input("snap", "value"), Input("expiry", "value"),
)
def draw_smile(path, idx):
    dash = "-"
    if not path:
        return empty_figure(NO_DATA), dash, dash, dash, dash, dash
    clean, slices, kr = build(path)
    if not slices or idx is None or int(idx) >= len(slices):
        return empty_figure(NO_DATA), dash, dash, dash, dash, dash

    T, params = slices[int(idx)]
    s = clean[np.isclose(clean["T"], T)].sort_values("log_m")
    kmin, kmax = kr[T]
    k = np.linspace(kmin, kmax, 250)

    fit_pts = np.sqrt(svi(s["log_m"].to_numpy(), *params) / T)
    rmse = np.sqrt(np.mean((fit_pts - s["iv_mid"].to_numpy()) ** 2)) * 100
    g = butterfly_min_on_range(params, kr[T])
    atm = np.sqrt(float(svi(0.0, *params)) / T) * 100

    fig = go.Figure()
    fig.add_scatter(
        x=s["log_m"], y=s["iv_mid"] * 100, mode="markers", name="Market mid (bid/ask bars)",
        marker=dict(color=C.BLUE, size=9, line=dict(color=C.BLACK, width=1.5)),
        error_y=dict(type="data", symmetric=False, color="rgba(61,139,255,0.45)", thickness=1.5,
                     array=((s["iv_ask"] - s["iv_mid"]) * 100).tolist(),
                     arrayminus=((s["iv_mid"] - s["iv_bid"]) * 100).tolist()),
        hovertemplate="k=%{x:.3f}<br>mid IV=%{y:.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=s["log_m"], y=s["iv_ref"] * 100, mode="markers", name="Deribit mark IV",
        marker=dict(color=C.YELLOW, symbol="x", size=8),
        hovertemplate="k=%{x:.3f}<br>mark IV=%{y:.2f}%<extra></extra>",
    )
    fig.add_scatter(
        x=k, y=np.sqrt(svi(k, *params) / T) * 100, mode="lines", name="SVI fit",
        line=dict(color=C.ORANGE, width=3.5, shape="spline"),
        hovertemplate="k=%{x:.3f}<br>SVI IV=%{y:.2f}%<extra></extra>",
    )
    fig.add_vline(x=0, line=dict(color="rgba(255,153,0,0.35)", dash="dot", width=1))
    fig.update_layout(
        height=540,
        xaxis_title="ln(K/F)",
        yaxis_title="Implied vol (%)",
    )
    return (fig, f"{len(s)}", f"{s['underlying_price'].iloc[0]:,.0f}",
            f"{atm:.1f}%", f"{rmse:.2f}", f"{g:.3f}")


@app.callback(Output("surface-graph", "figure"), Input("snap", "value"))
def draw_surface(path):
    if not path:
        return empty_figure(NO_DATA)
    clean, slices, kr = build(path)
    if len(slices) < 2:
        return empty_figure("At least two fitted expiries are needed for a surface")

    Ts = np.linspace(slices[0][0], slices[-1][0], 70)
    k = np.linspace(-0.8, 0.8, 110)
    z = np.full((len(Ts), len(k)), np.nan)
    for i, T in enumerate(Ts):
        lo, hi = k_range_at(T, slices, kr)
        z[i] = np.where((k >= lo) & (k <= hi), surface_iv(k, T, slices), np.nan)

    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=k, y=Ts * 365, z=z * 100, opacity=0.96,
        colorbar=dict(title="IV (%)"),
        contours_z=dict(show=True, usecolormap=True, project_z=False, width=1),
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.25, roughness=0.6),
        hovertemplate="k=%{x:.2f}<br>%{y:.0f} days<br>IV=%{z:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter3d(
        x=clean["log_m"], y=clean["T"] * 365, z=clean["iv_mid"] * 100, mode="markers",
        marker=dict(size=2.2, color=C.TEXT, opacity=0.55), name="Market mid",
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=700, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        scene=dict(
            xaxis_title="ln(K/F)",
            yaxis_title="Maturity (days)",
            zaxis_title="IV (%)",
            camera=dict(eye=dict(x=1.7, y=-1.7, z=0.9)),
            aspectmode="manual", aspectratio=dict(x=1.2, y=1.4, z=0.7),
        ),
    )
    return fig

@app.callback(Output("skew-table","children"), Output("rrbf-graph", "figure"), Output("atm-graph", "figure"), Input("snap","value"))
def draw_skew(path):
    if not path:
        return html.P(NO_DATA), empty_figure(NO_DATA), empty_figure(NO_DATA)
    clean, slices, kr = build(path)
    if not slices:
        return html.P(NO_DATA), empty_figure(NO_DATA), empty_figure(NO_DATA)

    tab = smile_table(slices, kr)
    sym = np.where(tab["in_range"], "circle", "circle-open").tolist() 

    fig_rrbf = go.Figure()
    for col, name in [("rr25", "RR25"), ("bf25", "BF25")]:
        fig_rrbf.add_scatter(x=tab["days"], y=tab[col] * 100, name=name,
                             mode="lines+markers", marker=dict(symbol=sym, size=9))
    fig_rrbf.add_hline(y=0, line_dash="dot", line_width=1)
    fig_rrbf.update_xaxes(type="log", title="Days to expiry")
    fig_rrbf.update_yaxes(title="Vol points")
    fig_rrbf.update_layout(height=420, title="RR25 and BF25 by maturity",
                           legend=dict(orientation="h", y=-0.2, x=0))

    fig_atm = go.Figure(go.Scatter(
        x=tab["days"], y=tab["atm"] * 100, name="ATM",
        mode="lines+markers", marker=dict(symbol=sym, size=9)))
    fig_atm.update_xaxes(type="log", title="Days to expiry")
    fig_atm.update_yaxes(title="ATM implied vol (%)")
    fig_atm.update_layout(height=420, title="ATM vol term structure", showlegend=False)

    return skew_table(tab), fig_rrbf, fig_atm

@app.callback(Output("gex-graph", "figure"), Input("snap", "value"))
def draw_gex(path):
    
    if not path:
        return empty_figure(NO_DATA)
    
    df = load_raw(path)
    gex = GEX(df)
    
    if not gex:
        return empty_figure(NO_DATA)

    nearest = df[df["expiry"] > df["snapshot_ts"].iloc[0]]["expiry"].min()
    spot = df.loc[df["expiry"] == nearest, "underlying_price"].iloc[0]

    s = pd.Series(gex).sort_index()
    s = s[(s.index > spot * 0.7) & (s.index < spot * 1.3)]
    if s.empty:
        return empty_figure(NO_DATA)

    fig = go.Figure(go.Bar(
        x=s.index,
        y=s.values / 1e6,
        marker_color=np.where(s.values >= 0, "#3fb950", "#f85149"),
    ))
    fig.add_vline(x=spot, line_dash="dash", line_width=2,
                  annotation_text="Spot", annotation_position="top")
    fig.update_layout(
        height=480,
        title="Gamma exposure by strike",
        xaxis_title="Strike",
        yaxis_title="GEX ($M per 1% move)",
        showlegend=False,
    )
    return fig

@app.callback(
    Output("iv30", "children"), Output("rv-ctc", "children"),
    Output("rv-p", "children"), Output("vrp", "children"),
    Output("vrp-graph", "figure"),
    Input("snap", "value"), State("ccy", "value"),
)
def draw_vrp(path, ccy):
    empty = ("—", "—", "—", "—", empty_figure(NO_DATA))
    if not path:
        return empty
    clean, slices, _ = build(path)
    if not slices:
        return empty
    try:
        vrp, roll_close, roll_park = compute_vrp(slices, ccy, clean["snapshot_ts"].iloc[0])
    except Exception:
        return empty
    if np.isnan(vrp["iv"]):  
        return empty

    fig = go.Figure()
    fig.add_scatter(x=roll_close.index, y=roll_close, name="RV close-to-close")
    fig.add_scatter(x=roll_park.index, y=roll_park, name="RV Parkinson")
    fig.add_hline(y=vrp["iv"], line_dash="dot", annotation_text="IV30")
    fig.update_yaxes(tickformat=".0%", title="Annualized vol")
    fig.update_layout(
    height=480,
    title="IV30 vs rolling 30d realized vol",
    legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
    margin=dict(b=90),
)

    return (f"{vrp['iv']:.1%}", f"{vrp['rv_cc']:.1%}", f"{vrp['rv_pk']:.1%}",
            f"{vrp['vrp_cc']*100:+.1f} pts", fig)


if __name__ == "__main__":
    app.run(debug=True)