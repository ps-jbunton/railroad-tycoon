"""
Module following the plotly dash tutorials.
"""

from dash import Dash, dcc, html
import pandas as pd
import plotly.graph_objects as go


def build_fig():
    """
    Function for building the plotly figure object from the last run.
    """
    with open("./.mapbox_key", "r", encoding="utf-8") as f:
        mapbox_token = f.read()

    output_data_dir = "../sim/output/"

    # Import static object data.
    rail_terminal_locations = pd.read_parquet(
        output_data_dir + "rail_terminal_locations.parquet"
    )
    container_yard_locations = pd.read_parquet(
        output_data_dir + "container_yard_locations.parquet"
    )

    # Import dynamic object data.
    num_parallel_vehicles = 1
    parallel_vehicles_outputs = {}
    for i in range(num_parallel_vehicles):
        parallel_vehicles_outputs[i] = pd.read_parquet(
            output_data_dir + f"parallel_vehicle_{i}.parquet"
        )

    # Make figure dictionaries
    fig_dict = {"data": [], "layout": {}, "frames": []}

    # fill in most of layout
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0, "easing": "linear"},
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 10},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top",
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Hour:",
            "visible": True,
            "xanchor": "right",
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 10},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    # make initial data
    for vehicle_id, result in parallel_vehicles_outputs.items():
        data_dict = {
            "type": "scattermapbox",
            "lat": [result["LATITUDE"][0]],
            "lon": [result["LONGITUDE"][0]],
            "mode": "markers",
            "text": [f"Parallel Vehicle {vehicle_id}"],
            "marker": go.scattermapbox.Marker(
                {
                    "size": 40,
                    "symbol": "circle",
                    "color": "rgb(200, 0, 0)",
                    "opacity": 0.7,
                }
            ),
            "showlegend": False,
            "name": f"Parallel Vehicle {vehicle_id}",
        }
        fig_dict["data"].append(data_dict)

    # make sequence of frames
    for vehicle_id, result in parallel_vehicles_outputs.items():
        for _, row in result.iterrows():
            frame = {"data": [], "name": f"{row['TIME']}"}
            data_dict = {
                "type": "scattermapbox",
                "lat": [row["LATITUDE"]],
                "lon": [row["LONGITUDE"]],
                "mode": "markers",
                "text": [f"Parallel Vehicle {vehicle_id}"],
                "marker": go.scattermapbox.Marker(
                    {
                        "size": 40,
                        "symbol": "circle",
                        "color": "rgb(200, 0, 0)",
                        "opacity": 0.7,
                    }
                ),
                "showlegend": False,
                "name": f"Parallel Vehicle {vehicle_id}",
            }
            frame["data"].append(data_dict)

            fig_dict["frames"].append(frame)
            slider_step = {
                "args": [
                    [row["TIME"]],
                    {
                        "frame": {"duration": 300, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 300},
                    },
                ],
                "label": f"{row['TIME']:.2f}",
                "method": "animate",
            }
            sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    map_fig = go.Figure(fig_dict)

    lat_lines = []
    lon_lines = []

    for _, row in container_yard_locations.iterrows():
        lat_lines.append(rail_terminal_locations["LATITUDE"][1])
        lat_lines.append(row["LATITUDE"])
        lon_lines.append(rail_terminal_locations["LONGITUDE"][1])
        lon_lines.append(row["LONGITUDE"])

    map_fig.add_trace(
        go.Scattermapbox(
            name="",
            mode="lines",
            lat=lat_lines,
            lon=lon_lines,
            showlegend=False,
        )
    )

    map_fig.add_trace(
        go.Scattermapbox(
            name="",
            mode="lines",
            lat=[lat for lat in rail_terminal_locations["LATITUDE"]],
            lon=[lon for lon in rail_terminal_locations["LONGITUDE"]],
            showlegend=False,
        )
    )

    map_fig.add_trace(
        go.Scattermapbox(
            lat=container_yard_locations["LATITUDE"],
            lon=container_yard_locations["LONGITUDE"],
            mode="markers",
            text=container_yard_locations["NAME"],
            marker=go.scattermapbox.Marker(size=10),
            showlegend=False,
            name="Container Yard",
        ),
    )

    map_fig.add_trace(
        go.Scattermapbox(
            lat=rail_terminal_locations["LATITUDE"],
            lon=rail_terminal_locations["LONGITUDE"],
            mode="markers",
            text=rail_terminal_locations["NAME"],
            marker=go.scattermapbox.Marker(size=15),
            showlegend=False,
            name="Rail Terminal",
        ),
    )

    map_fig.update_mapboxes(
        accesstoken=mapbox_token,
        center=dict(lat=31.97358, lon=-82.48189),
        style="outdoors",
        zoom=6.5,
    )
    map_fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return map_fig


if __name__ == "__main__":
    app = Dash(__name__)
    fig = build_fig()
    app.layout = html.Div([dcc.Graph(figure=fig)])
    app.run_server(debug=True)
