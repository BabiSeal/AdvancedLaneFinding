import numpy as np
import glob
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


def calibrate_camera(image_shape):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        else:
            print("Corners not found for", fname)

    img_size = (image_shape[1], image_shape[0])
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about
    # rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("camera_dist_pickle.p", "wb"))

    return

# Compute the Perspective Matrix


def compute_perspective_matrixes(straight_line_img_name):
    img_straight = cv2.imread(straight_line_img_name)
    img_size = (img_straight.shape[1], img_straight.shape[0])
    src = np.float32(
        [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def undistort_image(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


def threshold_pipeline(img, s_thresh=(170, 255), sx_thresh=(30, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:, :, 0]
    l_channel = hsv[:, :, 1]
    s_channel = hsv[:, :, 2]

    # Grayscale image
    # NOTE: we already saw that standard grayscaling lost color information for the lane lines
    # Explore gradients in other colors spaces / color channels to see what
    # might work better
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    # Absolute x derivative to accentuate lines away from horizontal
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) &
             (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can
    # see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    return color_binary, combined_binary


def warp_binary_image(img, M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)
    return warped


def pre_process_image(img, M, mtx, dist):

    # Undistort Image
    undist = undistort_image(img, mtx, dist)

    # Create binary image
    color_binary, combined_binary = threshold_pipeline(undist)

    # Apply perspective transform
    binary_warped = warp_binary_image(combined_binary, M)

    return binary_warped


def compute_lane_hist_bases_and_midpoints(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(
        binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return midpoint, leftx_base, rightx_base


def compute_left_and_right_fit(binary_warped,
                               prev_left_fit=None, prev_right_fit=None,
                               redetect=True, doplot=False):

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Set the width of the windows +/- margin
    margin = 80
    # Set minimum number of pixels found to recenter window
    minpix = 40

    if redetect == True:
        midpoint, leftx_base, rightx_base = compute_lane_hist_bases_and_midpoints(
            binary_warped)

        # Choose the number of sliding windows
        nwindows = 9

        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)

        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Step through the windows one by one
        for window in range(nwindows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean
        # position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

    else:
        left_lane_inds = ((nonzerox > (prev_left_fit[0] * (nonzeroy**2) + prev_left_fit[1] * nonzeroy + prev_left_fit[2] - margin)) & (
            nonzerox < (prev_left_fit[0] * (nonzeroy**2) + prev_left_fit[1] * nonzeroy + prev_left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (prev_right_fit[0] * (nonzeroy**2) + prev_right_fit[1] * nonzeroy + prev_right_fit[2] - margin)) & (
            nonzerox < (prev_right_fit[0] * (nonzeroy**2) + prev_right_fit[1] * nonzeroy + prev_right_fit[2] + margin)))

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if (len(leftx) == 0) or (len(lefty) == 0):
        return None, None

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # if we are asked to plot output
    if (doplot):

        # Create an output image to draw on and  visualize the result
        out_img = np.dstack(
            (binary_warped, binary_warped, binary_warped)) * 255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[
            left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[
            right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, binary_warped.shape[1])
        plt.ylim(binary_warped.shape[0], 0)

    return left_fit, right_fit


def compute_curvature(binary_warped, left_fit, right_fit):

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[
                     1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[
                      1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])

    # Now our radius of curvature is in meters
    #print(left_curverad, 'm', right_curverad, 'm')
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad


def compute_car_offset_from_center(img, left_lane_pixel, right_lane_pixel):

    xm_per_pix = 3.7 / 700  # meteres per pixel in x dimension
    screen_middle_pixel = img.shape[1] / 2
    car_middle_pixel = int((right_lane_pixel + left_lane_pixel) / 2)
    pixels_off_center = screen_middle_pixel - car_middle_pixel
    meters_off_center = xm_per_pix * pixels_off_center
    return meters_off_center


def build_result_image(warped, undist, left_fit, right_fit, Minv):
   # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    yvals = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0] * yvals**2 + left_fit[1] * yvals + left_fit[2]
    right_fitx = right_fit[0] * yvals**2 + right_fit[1] * yvals + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array(
        [np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective
    # matrix (Minv)
    newwarp = cv2.warpPerspective(
        color_warp, Minv, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    plt.imshow(result)
    left_curverad, right_curverad = compute_curvature(
        warped, left_fit, right_fit)
    avg_curverad = int((left_curverad + right_curverad) / 2)

    y_max = warped.shape[0]
    left_lane_pixel = left_fit[0] * y_max**2 + \
        left_fit[1] * y_max + left_fit[2]
    right_lane_pixel = right_fit[0] * y_max**2 + \
        right_fit[1] * y_max + right_fit[2]

    offset_from_center = compute_car_offset_from_center(
        warped, left_lane_pixel, right_lane_pixel)

    # Add text to plot
    text = "Radius of Curvature = {} m,\n Offset = {:.2f} m".format(
        avg_curverad, offset_from_center)
    plt.text(600, 100, text, horizontalalignment='center',
             verticalalignment='center', color='white')

    # Return result image
    return result, left_curverad, right_curverad, offset_from_center

'''
Class to keep state associated with each line.
'''


class Line():

    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

Left = Line()
Right = Line()

    '''
   Initializations for Camera Calibration and Perspective Matrixes
   '''
# Calibrate Camera for correcting distortion
img = cv2.imread('test_images/test1.jpg')
calibrate_camera(img.shape)
dist_pickle = pickle.load(open("camera_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Compute Perspective Images
M, Minv = compute_perspective_matrixes('test_images/straight_lines1.jpg')


'''
Discover the left and right lanes on this image and overay them on original image.
Invoked per frame/image on the video
'''


def process_image(img):

    alpha = 0.8

    # Pre-process the image to get an undistored, thresholded, binary, perspective
    # transformed image
    binary_warped = pre_process_image(img, M, mtx, dist)

    # Fit left and right lines
    if (Left.detected == False) or (Right.detected == False):
        Left.current_fit, Right.current_fit = compute_left_and_right_fit(
            binary_warped)
    else:
        Left.current_fit, Right.current_fit = compute_left_and_right_fit(binary_warped,
                                                                         Left.best_fit, Right.best_fit,
                                                                         False, False)

    # Smooth and process lines
    #

    yvals = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    # Process left line
    if Left.current_fit != None:

        if Left.best_fit == None:
            Left.best_fit = Left.current_fit

        # Smooth the xvals over previous fit and current-fit
        current_leftx = Left.current_fit[
            0] * yvals**2 + Left.current_fit[1] * yvals + Left.current_fit[2]
        last_leftx = Left.best_fit[0] * yvals**2 + \
            Left.best_fit[1] * yvals + Left.best_fit[2]

        # Update the x based on the moving average
        leftx = (alpha * last_leftx) + (1 - alpha) * current_leftx

        # Recompute the best fit coefficients
        Left.best_fit = np.polyfit(yvals, leftx, 2)

        # Found left lane
        Left.detected = True
    else:
        Left.detected = False

    # Process right line
    if Right.current_fit != None:

        if Right.best_fit == None:
            Right.best_fit = Right.current_fit

        # Smooth the xvals over previous fit and current-fit
        current_rightx = Right.current_fit[
            0] * yvals**2 + Right.current_fit[1] * yvals + Right.current_fit[2]
        last_rightx = Right.best_fit[0] * yvals**2 + \
            Right.best_fit[1] * yvals + Right.best_fit[2]

        # Update the x based on the moving average
        rightx = (alpha * last_rightx) + (1 - alpha) * current_rightx

        # Recompute the best fit coefficients
        Right.best_fit = np.polyfit(yvals, rightx, 2)

        # Found Right line
        Right.detected = True
    else:
        Right.detected = False

    # Compute curvature, draw lane boundaries and warp back to orignial image
    result, Left.radius_of_curvature, Right.radius_of_curvature, offset_from_center = build_result_image(
        binary_warped,
        img,
        Left.best_fit,
        Right.best_fit,
        Minv)
    # Print radius of curvature on video
    cv2.putText(result, 'Radius of Curvature {}(m)'.format(int((Left.radius_of_curvature + Right.radius_of_curvature) / 2)), (120, 140),
                fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)

    # Print distance from center on video
    if offset_from_center < 0:
        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(abs(offset_from_center)), (100, 80),
                    fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)
    else:
        cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(offset_from_center), (100, 80),
                    fontFace=16, fontScale=2, color=(255, 255, 255), thickness=2)

    # Return result image
    return result
