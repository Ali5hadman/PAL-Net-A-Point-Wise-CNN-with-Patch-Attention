from scipy.spatial import cKDTree
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm

def get_patches(X_new, AproximatedLandmarks, PointsPerPatch = 1000, reference_point = (0, 0, 0)):

    X = X_new
    # predictions = fix_prediction(X_new, np.array(AproximatedLandmarks, copy=None ))
    predictions = fix_prediction(X_new, np.array(AproximatedLandmarks))

    # print("predictions", predictions.shape)
   
    #=======================================================
    face_heatmaps = []

    # Loop over your set of point clouds (assuming X and predictions have the same outer length)
    for i in range(len(X)):
        # Get the current point cloud (shape: N x 6)
        # point_cloud = np.array(X[i], copy=None)
        point_cloud = np.array(X[i])
        
        # Extract only the XYZ coordinates (first three columns) for the k-d tree
        pc_xyz = point_cloud[:, :3]
        
        # Build a k-d tree using the 3D coordinates
        
        kdtree = cKDTree(pc_xyz)

        # For storing the 500 nearest points for each chosen points
        sampled_points = []
        
        # Loop over each prediction for the current point cloud
        for j in range(len(predictions[i])):
            # Get the current chosen point (should be a 3D coordinate)
            chosen_point = predictions[i][j]
            
            # Query the k-d tree for the 500 closest points
            # 'distances' are the distances to these points (not used here)
            # 'indices' are the indices into 'point_cloud'
            distances, indices = kdtree.query(chosen_point, k= PointsPerPatch)
            
            # Retrieve the 500 nearest points, including their color information
            nearest_points = point_cloud[indices]  # shape: (500, 6)
            
            # Append the result for this chosen point
            sampled_points.append(nearest_points)
        
        # Optionally convert sampled_points to a NumPy array before appending to face_heatmaps
        face_heatmaps.append(np.array(sampled_points))

    # Convert the list of heatmaps to a NumPy array if needed
    face_heatmaps = np.array(face_heatmaps)

    # The final array with sampled point clouds including XYZ and RGB data
    X_fine = face_heatmaps

    #=======================================================
    # sort based on distances to the reference point
    X_fine_final=[]
    reference_point = np.array(reference_point).reshape(1, 1, 3)
    for i in range(len(X_fine)):
        point_cloud = X_fine[i]
        distances = np.linalg.norm(point_cloud[:,:,:3] - reference_point, axis=2)  # Shape: (200, 100000)
        sorted_indices = np.argsort(distances, axis=1)  # Shape: (200, 100000)
        X_sorted = np.take_along_axis(point_cloud, sorted_indices[..., None], axis=1)
        X_fine_final.append(X_sorted)

    X_fine = np.array(X_fine_final)
    
    return X_fine

###############################################################################################################################################
import open3d as o3d

def resample(faces_aligned,toSample, replace= True):
    # Replace this with your unevenly sampled point clouds

    # Replace this with your unevenly sampled point clouds
    uneven_point_clouds = faces_aligned

    # desired_num_points = 10000  # The desired number of points after resampling
    desired_num_points =  toSample
    # desired_num_points =  41000


    resampled_point_clouds = []
    for point_cloud in uneven_point_clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)

        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.01)
        downsampled_points = np.asarray(downsampled_pcd.points).astype(np.float32)  # Force float32

        sample_indices = np.random.choice(downsampled_points.shape[0], desired_num_points, replace=replace)
        resampled = downsampled_points[sample_indices].astype(np.float32)  # Extra safety
        resampled_point_clouds.append(resampled)


    return resampled_point_clouds



def get_patches_distance_based(X_new, AproximatedLandmarks, PointsPerPatch = 1000, reference_point = (0, 0, 0)):

    X = X_new
    predictions = fix_prediction(X_new, np.array(AproximatedLandmarks, copy=None ))
    # print("predictions", predictions.shape)
  
    
    #=======================================================
    # global_radius = max(mean_errors) + 5  # Precompute the constant radius
    global_radius = 20
    face_heatmaps = []

    # Loop over each point cloud and its corresponding predictions
    for point_cloud, pred_points in zip(X, predictions):
        # Build the k-d tree once per point cloud
        kdtree = cKDTree(point_cloud)
        
        # Query all prediction points at once
        indices_list = kdtree.query_ball_point(pred_points, global_radius)
        
        # Extract the points within the radius for each prediction
        all_radius = [point_cloud[indices] for indices in indices_list]
        
        # Resample as needed
        sampled = resample(all_radius, PointsPerPatch)
        face_heatmaps.append(sampled)

    X_fine = np.array(face_heatmaps).astype(np.float32)

    #=======================================================
    # sort based on distances to the reference point
    X_fine_final=[]
    reference_point = np.array(reference_point).reshape(1, 1, 3)
    for i in range(len(X_fine)):
        point_cloud = X_fine[i]
        distances = np.linalg.norm(point_cloud[:,:,:3] - reference_point, axis=2)  # Shape: (200, 100000)
        sorted_indices = np.argsort(distances, axis=1)  # Shape: (200, 100000)
        X_sorted = np.take_along_axis(point_cloud, sorted_indices[..., None], axis=1)
        X_fine_final.append(X_sorted)

    X_fine = np.array(X_fine_final)
    return X_fine





###############################################################################################################################################

def fix_prediction(X,predictions):
    for i in range(len(X)):

        # Define your point cloud as a numpy array (each row is a point)
        if type(X) == list:
            point_cloud = np.array(X[i])
        else:
            point_cloud = X[i, :, :3]

        # point_cloud = X[i,0,:,:]

        for j in range(len(predictions[i])):

            # Define the point in space to which you want to find the closest point
            target_point = predictions[i][j]

            # Calculate the Euclidean distances between the target point and all points in the cloud
            distances = cdist([target_point], point_cloud, 'euclidean')

            # Find the index of the closest point
            closest_point_index = np.argmin(distances)

            # Get the closest point from the point cloud
            closest_point = point_cloud[closest_point_index]

            # # Print the closest point
            # print("Target Point:", target_point)
            # print("Closest Point:", closest_point)
            predictions[i][j] = closest_point

    return predictions


# def fix_precition_average(X, Predictions, k=3):
#     for i in range(len(X)):
#         # Define your point cloud as a numpy array (each row is a point)

#         if type(X) == list:
#             point_cloud = np.array(X[i])
#         else:
#             point_cloud = X[i, :, :3]


#         for j in range(len(Predictions[i])):
#             # Define the point in space to which you want to find the closest points
#             target_point = Predictions[i][j]

#             # Calculate the Euclidean distances between the target point and all points in the cloud
#             distances = cdist([target_point], point_cloud, 'euclidean')

#             # Find the indices of the k closest points
#             closest_point_indices = np.argsort(distances)[0][:k]

#             # Get the coordinates of the k closest points
#             closest_points = point_cloud[closest_point_indices]

#             # Calculate the barycenter of the k closest points
#             barycenter = np.mean(closest_points, axis=0)

#             # Assign the barycenter as the new prediction
#             Predictions[i][j] = barycenter

#     return Predictions


import numpy as np
from scipy.spatial import cKDTree

def fix_precition_average(X, Predictions, k=3):
    """
    Replace each predicted point with the barycenter of its k nearest neighbors,
    using cKDTree for speed but falling back if `workers` isn’t supported.
    """
    new_preds = []
    is_list = isinstance(X, list)

    for i, preds in enumerate(Predictions):
        # load point cloud
        pc = np.array(X[i]) if is_list else X[i, :, :3]
        tree = cKDTree(pc)

        targets = np.asarray(preds)

        # try parallel workers first, fallback if not supported
        try:
            dists, idxs = tree.query(targets, k=k, workers=-1)
        except TypeError:
            # older scipy: no `workers` parameter
            dists, idxs = tree.query(targets, k=k)

        # normalize shape: if k==1 you get (M,), make it (M,1)
        if idxs.ndim == 1:
            idxs = idxs[:, None]

        # compute barycenters in bulk
        barycenters = pc[idxs].mean(axis=1)
        new_preds.append(barycenters)

    return new_preds


###############################################################################################################################################

def finalize_landmark(y_pre, keep_all):
    y=[]
    for i in range(len(y_pre)):
        landmarks = y_pre[i]
        landmarks_sorted = landmarks[[ 0, 11, 22, 33, 44, 46, 47, 48, 49,  1,  2,  3,  4,  5,  6,  7,  8,
                                9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27,
                                28, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45]]


        landmarks_to_remove = [13,14,15,16,32,33,34,35]
        if keep_all:
            y.append(landmarks_sorted)
        else: 
            y.append(np.delete(landmarks_sorted, landmarks_to_remove, axis=0))

    y= np.array(y)
    return y

###############################################################################################################################################

def compute_distance_error(gt_landmarks, predicted_landmarks):
    n_cases, n_landmarks, _ = gt_landmarks.shape
    
    gt_landmarks = finalize_landmark(gt_landmarks, True)
    predicted_landmarks = finalize_landmark(predicted_landmarks, True)

    # Initialize arrays to store pairwise distances
    gt_distances = np.zeros((n_cases, n_landmarks, n_landmarks))
    predicted_distances = np.zeros((n_cases, n_landmarks, n_landmarks))
    
    # Calculate pairwise distances for ground truth and predicted landmarks
    for i in range(n_landmarks):
        for j in range(n_landmarks):
            gt_distances[:, i, j] = np.linalg.norm(gt_landmarks[:, i, :] - gt_landmarks[:, j, :], axis=1)
            predicted_distances[:, i, j] = np.linalg.norm(predicted_landmarks[:, i, :] - predicted_landmarks[:, j, :], axis=1)
    
    # Compute absolute difference and average over all cases
    absolute_difference = np.abs(gt_distances - predicted_distances)
    average_error = np.mean(absolute_difference, axis=0)  # Shape (50, 50)
    
    return average_error


def compute_distance_error_std(gt_landmarks, predicted_landmarks):
    n_cases, n_landmarks, _ = gt_landmarks.shape
    
    # gt_landmarks = finalize_landmark(gt_landmarks, True)
    # predicted_landmarks = finalize_landmark(predicted_landmarks, True)

    # Initialize arrays to store pairwise distances
    gt_distances = np.zeros((n_cases, n_landmarks, n_landmarks))
    predicted_distances = np.zeros((n_cases, n_landmarks, n_landmarks))
    
    # Calculate pairwise distances for ground truth and predicted landmarks
    for i in range(n_landmarks):
        for j in range(n_landmarks):
            gt_distances[:, i, j] = np.linalg.norm(gt_landmarks[:, i, :] - gt_landmarks[:, j, :], axis=1)
            predicted_distances[:, i, j] = np.linalg.norm(predicted_landmarks[:, i, :] - predicted_landmarks[:, j, :], axis=1)
    
    # Compute absolute difference and average over all cases
    absolute_difference = np.abs(gt_distances - predicted_distances)
    std_error = np.std(absolute_difference, axis=0)  # Shape (50, 50)
    
    return std_error




def calculate_error(predicted_coords, actual_coords):
    errors = np.sqrt(np.sum((predicted_coords - actual_coords) ** 2, axis=1))
    return errors

def compute_pointwise_error(pred, y_true):

    all_error=[]
    predictions = []

    y_true_finalized = finalize_landmark(y_true, True)
    predictions_finalized = finalize_landmark(pred,  True)
    
    # y_true_finalized = y_true
    # predictions_finalized = pred
   
    for i in range(len(predictions_finalized)):
        errors = calculate_error(predictions_finalized[i], y_true_finalized[i])
        all_error.append(errors)
    all_error = np.array(all_error).transpose()

    mean_errors=[]
    for i in range(len(all_error)):
        mean_errors.append(all_error[i].mean())

    return np.array(mean_errors)



def compute_pointwise_error_std(pred, y_true):

    all_error=[]
    predictions = []

    # y_true_finalized = finalize_landmark(y_true, True)
    # predictions_finalized = finalize_landmark(pred,  True)
    
    y_true_finalized = y_true
    predictions_finalized = pred
   
    for i in range(len(predictions_finalized)):
        errors = calculate_error(predictions_finalized[i], y_true_finalized[i])
        all_error.append(errors)
    all_error = np.array(all_error).transpose()

    mean_errors=[]
    for i in range(len(all_error)):
        mean_errors.append(all_error[i].std())

    return np.array(mean_errors)





