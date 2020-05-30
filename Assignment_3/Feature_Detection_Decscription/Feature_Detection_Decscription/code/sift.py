import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.color import rgb2gray
   
def get_features(image, x, y, feature_width, scales=[1]):
    """
    In this function, you need to implement the SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint. 
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    
    # Co-ordinate list storing all the co-ordinates of feature points
    crd_list = list(zip(x, y)) 
    
    scale = scales[0]
    image = rgb2gray(image)
    
    # Constructing the horizontal and vertical gradiant matrix
    I_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, 3)
    I_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, 3)
    
    
    orientation_matrix = np.degrees(np.arctan2(I_y,I_x))
    magnitude_matrix = np.sqrt(I_x**2 + I_y**2)
    
    
    def fit_parabola(histogram):
        
        # Given a histogram this function fits a parabola with the maximum bin
        # .. and 2 other orientation bins nearest to it in terms of magnitude.
        
        # This function is used to find the DOMINANT ORIENTATION for a histogram
        
        # The function returns the (bin_no * 1) orientation vector 
        
        bin_wdt = 360/len(histogram)
            
        sorted_hist = np.sort(histogram)
        # We take the maximum 3 orientation bins
        # To fit the parabola we use the bin values(weighted magnitudes)
        # .. of those 3 bins as y co-ordinates
        
        max_y = sorted_hist[-1]
        max_y_2 = sorted_hist[-2]
        max_y_3 = sorted_hist[-3]
        
          
        # To fit the parabola we use the mid-points of the bins as x co-ordinates
        max_x = np.where(histogram == max_y)[0][0] * bin_wdt + bin_wdt/2
        max_x_2 = np.where(histogram == max_y_2)[0][0] * bin_wdt + bin_wdt/2
        max_x_3 = np.where(histogram == max_y_3)[0][0] * bin_wdt + bin_wdt/2
        
        
        A = np.array([[max_x**2, max_x, 1], 
                    [max_x_2**2, max_x_2, 1], 
                    [max_x_3**2, max_x_3, 1]]) 

        b = np.array([max_y, 
                    max_y_2, 
                    max_y_3]) 
        
        try:
            coeff = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return max_x
            
        return -coeff[1]/(2*coeff[0])
        
    
    
    def build_histogram(orientation_matrix, magnitude_matrix, num_bin):
        
        # this function builds the histogram with "num_bins" bins
        # .. for given orientation & magnitude matrix
        
        bin_wdt = 360/num_bin    # width of each bin of the histogram
        
        orientation = orientation_matrix.ravel()
        # The following code transforms the orientation to range [0, 360]
        for i in range(0, len(orientation)):
            while orientation[i]<0:
                orientation[i] = 360 + orientation[i]
            while orientation[i]>360:
                orientation[i] = orientation[i] - 360
                
        # applying Gaussian weight to the magnitude matrix
        magnitude = ndi.gaussian_filter(magnitude_matrix, sigma=1.5*scale)
        magnitude = magnitude_matrix.ravel()
        
        assert len(magnitude) == len(orientation), "magnitude and orientation lengths differ"
        
        histogram = np.zeros(num_bin)
        
        for i in range(0,len(orientation)):
                
            # identifying the histogram bin to which the orientation belongs
            if orientation[i] != 360:    
                hist_index = int(orientation[i] // bin_wdt)
            else:
                hist_index = bin_wdt-1
            
            # Next we use interpolation to compute the weights to put the 
            # .. magnitudes in the corresponding bins
            if hist_index not in [0,num_bin-1]:
            
                mid_1 =  hist_index*bin_wdt + bin_wdt/2
                
                if orientation[i] > mid_1:
                    
                    # The magnitude corresponding to the orientation is put in 2 bins weighted
                    # .. by the distance of it from the corresponding the bin centers
                    
                    wt = (1-(orientation[i]-mid_1)/bin_wdt)
                    histogram[hist_index] += magnitude[i] * wt
                    histogram[hist_index+1] += magnitude[i] * (1-wt) 


                elif orientation[i] < mid_1:
                    
                    # The magnitude corresponding to the orientation is put in 2 bins weighted
                    # .. by the distance of it from the corresponding the bin centers
                    
                    wt = (1-(mid_1-orientation[i])/bin_wdt)
                    histogram[hist_index] += magnitude[i] * wt
                    histogram[hist_index-1] += magnitude[i] * (1-wt)
                    
                    
                else:
                    histogram[hist_index] += magnitude[i]
            
            else:        
            # appending the Gaussian weighted magnitude to that histogram bin
                histogram[hist_index] += magnitude[i]
            
        return histogram   


    # the list f_v stores all the 128x1 feature vectors for all the keypoints
    fv = []
    
    
    for c in crd_list:
        x = c[1]
        y = c[0]
        
        orientation = orientation_matrix[int(x-(feature_width/2)+1) : int(x+(feature_width/2)+1),
                                         int(y-(feature_width/2)+1) : int(y+(feature_width/2)+1)]
        
        magnitude = magnitude_matrix[int(x-(feature_width/2)+1) : int(x+(feature_width/2)+1),
                                     int(y-(feature_width/2)+1) : int(y+(feature_width/2)+1)]
        
        assert magnitude.shape == (feature_width, feature_width), "inconsistent magnitude shape"
        assert orientation.shape == (feature_width, feature_width), "inconsistent orientation shape"
        
        
        # =======================================
        # Orientation Assignment to each keypoint
        # =======================================
        
        
        # First we construct a histogram out of the feature_width x feature_width window
        # .. around each keypoint to find the dominant orientation
        
        num_bin = 36
        bin_wdt = 360/num_bin 
        primary_hist = build_histogram(orientation, magnitude, num_bin=num_bin)
        
        sorted_hist = np.sort(primary_hist)
        # We compare the maximum 3 orientation bins to identify if there are multiple peaks
        
        max_peak = sorted_hist[-1]
        max_peak_2 = sorted_hist[-2]
        
       # TODO: compare 3rd max also
        if max_peak_2 < 0.8 * max_peak:
            dominant_orientation = np.where(primary_hist == max_peak)[0][0]* bin_wdt + bin_wdt/2
        else:
            dominant_orientation = fit_parabola(primary_hist)     
        
        # To make the algorithm ROTATION INVARIANT we rotate the neighbourhood of the 
        # .. keypoint by the angle of the dominant orientation
        # .. i.e. we have to subtract the dominant orientation from escah of the orientations
        # Also we keep the dominant orientation as the the orientation for the keypoint
            
        orientation_rotated = orientation - dominant_orientation
        orientation_rotated[int((feature_width/2)-1), int((feature_width/2)-1)] = dominant_orientation
        
        orientation = orientation_rotated.ravel()
        # The following code transforms the orientation to range [0, 360]
        for i in range(0, len(orientation)):
            while orientation[i]<0:
                orientation[i] = 360 + orientation[i]
                
        orientation_rotated = np.reshape(orientation, (feature_width, feature_width))
        
        
        # ===================================
        # Local Feature Descriptor generation
        # ===================================
        
        
        # Next we divide this "feature_width x feature_width" window into "4 x 4" grids
        # For each of the 16 grids we construct the histogram with 8 bins
        
        grid_size = 4     # we divide the window into 4x4 grids  
        num_bin = 8      #  No. of bins in the histogram
        num_grid = int(feature_width/grid_size) # 4 for feature_width = 16
    
        fv_grid = []
        for i in range(0, num_grid): 
          for j in range(0, num_grid): 
            
            top, left = i*num_grid, j*num_grid
            bottom, right = i*num_grid+num_grid-1, i*num_grid+num_grid-1
            
            # grid_orientation stores is the part of the orientation matrix for the i,j -th grid
            grid_orientation = orientation_rotated[top:bottom, left:right]
            # grid_magnitude stores is the part of the magnitude matrix for the i,j -th grid
            grid_magnitude = magnitude[top:bottom, left:right]
            
            
            # Normalizaton
            un_norm = build_histogram(grid_orientation, grid_magnitude, num_bin=num_bin)
            if np.sum(un_norm) == 0:
                f_vect_normed = un_norm
            else:
                f_vect_normed = un_norm/np.sum(un_norm)
            
            fv_grid.append(f_vect_normed)
            
        temp = np.array(fv_grid).flatten()
        temp_1 = temp.tolist()
        
        fv.append(temp_1)
            
    return fv
