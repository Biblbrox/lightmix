import argparse
import glob
import io
import re
from os.path import basename, dirname, join

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars as pl
from dash import Dash, Input, Output, callback, clientside_callback, dcc, html
from plotly.subplots import make_subplots


def parse_logs(root_dir):
    """Parse all log files into a DataFrame."""
    epoch_regex = re.compile(r"epoch-(\d+)")
    rows = []

    for file_path in glob.glob(join(root_dir, "*", "**/*.log"), recursive=True):
        exp = file_path.split("/")[-4]
        split = "train" if "train" in file_path else "valid"

        match = epoch_regex.search(basename(dirname(file_path)))
        if not match:
            continue

        df = pl.read_csv(file_path, has_header=False, use_pyarrow=True)
        rows.append({
            "experiment": exp,
            "split": split,
            "metric": basename(file_path).replace(".log", ""),
            "epoch": int(match.group(1)),
            "value": df["column_1"].mean(),
            "err": df["column_1"].std(),
        })

    return pl.DataFrame(rows).sort(["experiment", "metric", "epoch"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Experiment tracking")
    parser.add_argument(
        "-d", dest="d", help="Path to experiment's root folder", type=str, required=True
    )
    args = parser.parse_args()

    df = parse_logs(args.d)
    experiments = df["experiment"].unique().to_list().sort()
    app = Dash(external_stylesheets=[dbc.themes.DARKLY])
    PALETTE = [
        "#636EFA",
        "#EF553B",
        "#00CC96",
        "#AB63FA",
        "#FFA15A",
        "#19D3F3",
        "#FF6692",
        "#B6E880",
    ]

    clientside_callback(
        """function(dark) {
            const link = document.querySelector('link[rel=stylesheet]');
            link.href = dark ? '%s' : '%s';
            return null;
        }"""
        % (dbc.themes.DARKLY, dbc.themes.FLATLY),
        Output("theme-store", "data"),
        Input("theme-switch", "value"),
    )

    app.layout = dbc.Container(
        [
            dcc.Interval(id="poll", interval=5_000, n_intervals=0),
            dcc.Store(id="df-store"),
            dcc.Store(id="theme-store"),
            dbc.Row(
                dbc.Col([
                    html.H2("Experiment Dashboard", className="d-inline me-3"),
                    dbc.Switch(
                        id="theme-switch",
                        label="Dark",
                        value=True,
                        className="d-inline-flex",
                    ),
                ]),
                class_name="align-items-center py-2 border-bottom mb-3",
            ),
            dbc.Row([
                dbc.Col(
                    [
                        html.H4("Experiments", className="mb-3"),
                        dbc.Checklist(
                            id="experiment-selector",
                            options=[
                                {
                                    "label": html.Span([
                                        html.Span(
                                            style={
                                                "width": "10px",
                                                "height": "10px",
                                                "borderRadius": "50%",
                                                "display": "inline-block",
                                                "marginRight": "8px",
                                                "backgroundColor": PALETTE[i % len(PALETTE)],
                                            }
                                        ),
                                        name,
                                    ]),
                                    "value": name,
                                }
                                for i, name in enumerate(df["experiment"].unique())
                            ],
                            value=[df["experiment"][0]],
                            input_class_name="btn-check",
                            label_class_name="btn btn-outline-secondary w-100 text-start mb-1",
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(html.Div(id="plots-container"), width=10),
            ]),
        ],
        fluid=True,
    )

    @callback(
        Output("df-store", "data"),
        Output("experiment-selector", "options"),
        Input("poll", "n_intervals"),
    )
    def refresh_store(n):
        fresh = parse_logs(args.d)
        exps = fresh["experiment"].unique().to_list()
        return fresh.write_json(), [
            {
                "label": html.Span([
                    html.Span(
                        style={
                            "width": "10px",
                            "height": "10px",
                            "borderRadius": "50%",
                            "display": "inline-block",
                            "marginRight": "8px",
                            "backgroundColor": PALETTE[i % len(PALETTE)],
                        }
                    ),
                    name,
                ]),
                "value": name,
            }
            for i, name in enumerate(exps)
        ]

    @callback(
        Output("plots-container", "children"),
        Input("experiment-selector", "value"),
        Input("theme-switch", "value"),
        Input("df-store", "data"),
    )
    def update_plots(selected, is_dark, store_data):
        if not selected:
            return []
        template = "plotly_dark" if is_dark else "plotly_white"
        df = pl.read_json(io.StringIO(store_data)) if store_data else pl.DataFrame()
        filtered = df.filter(pl.col("experiment").is_in(selected))
        exp_color = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(selected)}

        cards = []
        for metric in sorted(filtered["metric"].unique()):
            metric_df = filtered.filter(pl.col("metric") == metric)
            fig = make_subplots(cols=2, subplot_titles=["train", "valid"])

            for col, split in enumerate(("train", "valid"), start=1):
                split_df = metric_df.filter(pl.col("split") == split)
                for exp in selected:
                    exp_df = split_df.filter(pl.col("experiment") == exp)
                    if exp_df.is_empty():
                        continue
                    fig.add_trace(
                        go.Scatter(
                            x=exp_df["epoch"].to_list(),
                            y=exp_df["value"].to_list(),
                            error_y=dict(type="data", array=exp_df["err"].to_list(), visible=True),
                            name=exp,
                            line=dict(color=exp_color[exp]),
                            legendgroup=exp,
                            showlegend=(col == 1),
                        ),
                        col=col,
                        row=1,
                    )

            fig.update_layout(template=template)
            cards.append(
                dbc.Card(
                    [
                        dbc.CardHeader(html.H3(metric, className="mb-0")),
                        dbc.CardBody(dcc.Graph(figure=fig)),
                    ],
                    className="mb-4",
                )
            )

        return cards

    # app.run(debug=True)
    app.run(debug=False, dev_tools_ui=False)
