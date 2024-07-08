import numpy as np
import math


def f_computed(pts1, pts2):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2))
    #Output: fundamental matrix (numpy array of shape (3, 3))

    ctd_1 = np.mean(pts1, axis=0)
    ctd_2 = np.mean(pts2, axis=0)
    # Compute mean distance of all points to the centroids.
    distances_1 = []
    distances_2 = []
    for i in range(pts1.shape[0]):
        distances_1.append(math.dist(pts1[i], ctd_1))
        distances_2.append(math.dist(pts2[i], ctd_2))

    distances_1 = np.array(distances_1)
    distances_2 = np.array(distances_2)
    mean_dist_1 = np.mean(distances_1, axis=0)
    mean_dist_2 = np.mean(distances_2, axis=0)

    # Apply translation to the points to bring centroid to origin.
    ones = np.ones((pts1.shape[0], 1), dtype=np.int32)
    pts1 = np.hstack((pts1, ones))
    pts2 = np.hstack((pts2, ones))
    T1 = Construct_Transformation_Matrix(mean_dist_1, -ctd_1[0], -ctd_1[1])
    T2 = Construct_Transformation_Matrix(mean_dist_2, -ctd_2[0], -ctd_2[1])
    pts1_new = (T1 @ pts1.T) # 3 x N
    pts2_new = (T2 @ pts2.T) # 3 x N

    A = np.zeros((pts1_new.shape[1], 9))
    for i in range(pts1_new.shape[1]):
        x1 = np.array([pts1_new[:, i]]).reshape((3, 1))
        x2 = np.array([pts2_new[:, i]]).reshape((3, 1))
        row = x2 @ x1.T
        row = np.reshape(row, (1, 9))
        A[i] = row

    u, sig, vh = np.linalg.svd(A)
    v = vh.T
    F_est = v[:, -1]
    F_est = np.reshape(F_est, (3, 3))
    # Now find SVD for F_est and make it a rank 2 matrix by setting 3rd eigen value to 0 and recalculate F.
    u, sig, vh = np.linalg.svd(F_est)
    sig[2] = 0
    F = u @ np.diag(sig) @ vh
    # Denormalize the matrix F with the translation matrix that we applied for 2 images' points.
    F_mat = T2.T @ F @ T1
    # Since the last coordinate can be voided as a scaling factor, we can normalize the fundamental matrix with it.
    # (Eight point Algorithm)
    F_mat = F_mat * (1 / F_mat[2, 2])
    return F_mat
    
def findfundamentalmatrix(matchedpoints , num_trials = 1000, threshold = 0.01):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2)).
    #Output: fundamental matrix (numpy array of shape (3, 3)).

    # Set parameters and initialize variables.
    adaptive = True
    pts1 = matchedpoints[0] # First set of points
    pts2 = matchedpoints[1]  # Second set of points
    size = pts1.shape[0]
    s = 8  # no of point pairs required to compute fundamental matrix
    t = threshold  # Epsilon threshold
    n = num_trials

    e = 0.5  # Assumption Outlier Ratio => no. of outliers / no. of points
    # Adaptively determining value of trials/ iterations - N
    if adaptive == True:
        p = 0.95  # Required probability of Success
        n = int(np.log(1-p) / np.log(1 - (pow(1-e, s))))
    else:
        n = num_trials

    F_arr = []  # Array to store Fundamental matrices for each set of sample points.
    inliers_count_arr = []  # Array to store no. of inliers for corresponding Fundamental Matrix.

    # RANSAC Loop
    for i in range(n):
        # Sample 8 points from correspondences pts1 and pts2.
        ## Generate 8 random unique integers between 0 and no. of correspondences for indices.
        indices = sorted(np.random.choice(size, s, replace=False))
        train_pts1 = pts1[indices]
        train_pts2 = pts2[indices]
        ## Compute Fundamental Matrix using the function written above
        F = f_computed(train_pts1, train_pts2)
        F_arr.append(F)  # Append to list of fundamental matrices

        ## Calculate number of inliers and outliers using Fundamental matrix.
        ### Remember: pts2.T @ F @ pts1 < t ~ 0 # Where t is the threshold.
        test_pts1 = np.delete(pts1, indices, axis=0)
        test_pts2 = np.delete(pts2, indices, axis=0)
        ones = np.ones((test_pts1.shape[0], 1), dtype=np.int32)
        test_pts1 = np.hstack((test_pts1, ones))
        test_pts2 = np.hstack((test_pts2, ones))
        # ones = np.ones((pts1.shape[0], 1), dtype=np.int32)
        # test_pts1 = np.hstack((pts1, ones))
        # test_pts2 = np.hstack((pts2, ones))

        ### Find number of inliers in pts2_c => How many lie inside the threshold t.
        inliers_count = 8

        #### Run a loop against all the entries of pts2 and compare each point with pts2_c and check if it lies in the
        #### desired range of pts2 +- t (threshold).
        for j in range(test_pts1.shape[0]):
            #### loss error
            error = abs(test_pts2[j] @ F @ test_pts1[j])
            # e1 = test_pts2[j] @ F
            #
            # a = e1[0] * test_pts1[j, 0]
            # b = e1[1] * test_pts1[j, 1]
            # c = e1[2] * test_pts1[j, 2]
            #
            # error = np.sqrt(a*a + b*b + c*c)

            if error <= t:
                inliers_count += 1
        # Save the number of inliers in this model based on the loss and threshold above.
        inliers_count_arr.append(inliers_count)

    # Return the best fundamental matrix which fit most number of points as inliers.
    F_best = F_arr[np.argmax(inliers_count_arr)]
    return F_best
