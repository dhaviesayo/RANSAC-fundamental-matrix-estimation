import numpy as np



def findfundamentalmatrix(matchedpoints , num_trials = 1000, threshold = 0.01):
    #Input: two lists of corresponding keypoints (numpy arrays of shape (N, 2)).
    #Output: fundamental matrix (numpy array of shape (3, 3)).

    # Set parameters and initialize variables.
    adaptive = False
    pts1 = matchedpoints[0] # First set of points
    pts2 = matchedpoints[1]  # Second set of points
    size = pts1.shape[0]
    s = 8  # no of point pairs required to compute fundamental matrix
    t = threshold  # Epsilon threshold
    N = num_trials

    e = 0.5  # Assumption Outlier Ratio => no. of outliers / no. of points
    # Adaptively determining value of trials/ iterations - N
    if adaptive:
        p = 0.95  # Required probability of Success
        N = np.log(1-p) / np.log(1 - (pow(1-e, s)))

    F_arr = []  # Array to store Fundamental matrices for each set of sample points.
    inliers_count_arr = []  # Array to store no. of inliers for corresponding Fundamental Matrix.

    # RANSAC Loop
    for i in range(int(N)):
        # Sample 8 points from correspondences pts1 and pts2.
        ## Generate 8 random unique integers between 0 and no. of correspondences for indices.
        indices = sorted(np.random.choice(size, s, replace=False))
        train_pts1 = pts1[indices]
        train_pts2 = pts2[indices]
        ## Compute Fundamental Matrix using the function written above
        F = findfundamentalmatrix(train_pts1, train_pts2)
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
