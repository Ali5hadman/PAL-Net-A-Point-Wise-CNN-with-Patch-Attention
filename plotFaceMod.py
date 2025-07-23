

import numpy as np
import plotly.graph_objects as go

face_num = 0


def plotFace(point_cloud,
         actual_landmarks,
            predicted_landmarks,
            rgb_colors=None,):
        

    if rgb_colors is None:
        rgb_colors = "blue"
    # # Replace with actual data
    # point_cloud = X_test_pre[face_num]
    # rgb_colors = colors_test[face_num]/255
    # actual_landmarks = y_test[face_num]
    # predicted_landmarks = preds_np[face_num]




    # Create scatter plot for the original point cloud with colors
    scatter_original = go.Scatter3d(
        x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2],
        mode='markers',
        marker=dict(size=0.5, color=rgb_colors, opacity=1),
        name='Point Cloud'
    )

    # Create scatter plot for actual landmarks (with hover labels)
    scatter_actual_landmarks = go.Scatter3d(
        x=actual_landmarks[:, 0], 
        y=actual_landmarks[:, 1], 
        z=actual_landmarks[:, 2], 
        mode='markers',
        marker=dict(size=2.5, color='green', symbol='circle', opacity=1),
        # hovertext=landmark_names,  # Shows name on hover
        hoverinfo="text",
        name='Actual Landmarks'
    )

    # Create scatter plot for predicted landmarks (with hover labels)
    scatter_predicted_landmarks = go.Scatter3d(
        x=predicted_landmarks[:, 0], 
        y=predicted_landmarks[:, 1], 
        z=predicted_landmarks[:, 2], 
        mode='markers',
        marker=dict(size=2.5, color='red', symbol='circle', opacity=1),
        # hovertext=landmark_names,  # Shows name on hover
        hoverinfo="text",
        name='Predicted Landmarks'
    )

    # Create the 3D figure
    fig = go.Figure(data=[scatter_original, scatter_actual_landmarks, scatter_predicted_landmarks])

    # Set layout options
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(backgroundcolor="white", showgrid=False),
            yaxis=dict(backgroundcolor="white", showgrid=False),
            zaxis=dict(backgroundcolor="white", showgrid=False)
        ),
        scene_camera=dict(eye=dict(x=1.25, y=1.25, z=1.25)),
        title='Point Cloud with Hover Landmarks',
        showlegend=True,
        width=1200,
        height=800,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Show the interactive 3D plot
    fig.show()

    # Save the figure as an HTML file
    # fig.write_html("point_cloud_landmarks.html")
