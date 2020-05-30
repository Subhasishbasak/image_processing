import cv2
import numpy as np
import matplotlib.pyplot as plt

# Additional imports 
from scipy import ndimage as ndi
from skimage.color import rgb2gray
from scipy import signal


def get_interest_points(image, feature_width):
    """
    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You can create additional interest point detector functions for 
    extra credit.

    If you're finding spurious interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful in this function in order to (a) suppress boundary interest
    points (where a feature wouldn't fit entirely in the image, anyway)
    or (b) scale the image filters being used. Or you can ignore it.

    By default you do not need to make scale and orientation invariant
    local features.

    The lecture slides and textbook are a bit vague on how to do the
    non-maximum suppression once you've thresholded the cornerness score.
    You are free to experiment. 

    Args:
    -   image: A numpy array of shape (m,n,c),
                image may be grayscale of color (your choice)
    -   feature_width: integer representing the local feature width in pixels.

    Returns:
    -   x: A numpy array of shape (N,) containing x-coordinates of interest points
    -   y: A numpy array of shape (N,) containing y-coordinates of interest points
    -   confidences (optional): numpy nd-array of dim (N,) containing the strength
            of each interest point
    -   scales (optional): A numpy array of shape (N,) containing the scale at each
            interest point
    -   orientations (optional): A numpy array of shape (N,) containing the orientation
            at each interest point
    """
    confidences, scales, orientations = None, None, None
    
 #   img_gray = ndi.gaussian_filter(img_gray, sigma=1)
    
    def horiz_grad(img):
        # Function for computing the horizontal gradiant
        
        kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
        return signal.convolve2d(img, kernel, mode='same')
    
    
    def vert_grad(img):
        # Function for computing the vertical gradiant
        
        kernel = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])
        return signal.convolve2d(img, kernel, mode='same')

    
    I_x = horiz_grad(image)
    I_y = vert_grad(image)
    
    Ixx = ndi.gaussian_filter(I_x**2, sigma=1)
    Ixy = ndi.gaussian_filter(I_y*I_x, sigma=1)
    Iyy = ndi.gaussian_filter(I_y**2, sigma=1)
    
    k = 0.06
    
    detA = Ixx * Iyy - Ixy ** 2
    traceA = Ixx + Iyy
        
    harris_response = detA - k * traceA ** 2
    
    r_max = np.max(harris_response)
    

    
    def neighbour_ckeck(harris_response, window = 1):
        # This function compares a feature point with its neighbours
        # Given a window size it considers all neighbours in that window of a feature point
        # if a feature pt has greater response strength than all its neighbours it returns
        
        def check(list1, val): 
        # function for comparing a value with a list
        
            return(all(x < val for x in list1)) 
        
        x_1 = []
        y_1 = []
        harris_tuple = []
        
        for x in range(0,harris_response.shape[0]):
            
            for y in range(0,harris_response.shape[1]):
                
                # The function only considers those response strengths which are 
                # r > 0.0009 * r_max
                
                if harris_response[x,y] > 0.009 * r_max:
                    
                    neighbours = []
                    
                    for i in range(1, window+1):
                        
                        if x+i in range(0, harris_response.shape[0]):
                            neighbours.append(harris_response[x+i,y])
                            for j in range(1,i+1):
                                if y+j in range(0, harris_response.shape[1]):
                                    neighbours.append(harris_response[x+i, y+j])
                                if y-j in range(0, harris_response.shape[1]):
                                    neighbours.append(harris_response[x+i, y-j])
                        
                        if x-i in range(0, harris_response.shape[0]):
                            neighbours.append(harris_response[x-i,y])
                            for j in range(1,i+1):
                                if y+j in range(0, harris_response.shape[1]):
                                    neighbours.append(harris_response[x-i, y+j])
                                if y-j in range(0, harris_response.shape[1]):
                                    neighbours.append(harris_response[x-i, y-j])
                        
                        if y+i in range(0, harris_response.shape[1]):
                            neighbours.append(harris_response[x,y+i])
                            for j in range(1,i):
                                if x+j in range(0, harris_response.shape[0]):
                                    neighbours.append(harris_response[x+j,y+i])
                                if x-j in range(0, harris_response.shape[0]):
                                    neighbours.append(harris_response[x-j,y+i])
                        
                        if y-i in range(0, harris_response.shape[1]):
                            neighbours.append(harris_response[x,y-i])
                            for j in range(1,i):
                                if x+j in range(0, harris_response.shape[0]):
                                    neighbours.append(harris_response[x+j,y-i])
                                if x-j in range(0, harris_response.shape[0]):
                                    neighbours.append(harris_response[x-j,y-i])
                
                if check(neighbours, harris_response[x,y]):

                    x_1.append(x)
                    y_1.append(y)
                    harris_tuple.append([harris_response[x,y], x, y])
        
        return harris_tuple
        
    # The following list harris_tuple stores a tuple of the following :
    # [response strength, x_co-ordinate, y_co-ordinate]
    
    window = 1
    harris_tuple = neighbour_ckeck(harris_response, window = window)
    
    print("No. of feature pts. after Neighbourhood check", len(harris_tuple))
    print("Window size used for Neighbourhood check", (window*2+1))
    
    
    
    def check_feature_width(harris_tuple, feature_width, im_ht, im_wt):
        
        # This function checks whether the selected feature points satisfies boundary cond.
        # This restricts feature points to fall within feature width of the boundary
        
        remove_list = []
        for i in harris_tuple:
            if i[1] not in list(range(feature_width//2, im_ht - feature_width//2)) or i[2] not in list(range(feature_width//2, im_wt - feature_width//2)):
                
                remove_list.append(harris_tuple.index(i))
#                print("removing : ", i)
                
        harris_tuple = np.array(harris_tuple) 
        harris_tuple = np.delete(np.array(harris_tuple), remove_list, axis = 0)
        
        
        print("No. of feature pts. after feature width contrain", len(harris_tuple))
        
        return harris_tuple.tolist()
    
    
    harris_tuple = check_feature_width(harris_tuple, feature_width, image.shape[0], image.shape[1])
    
    
    
    def Sort_Tuple(tup, k, reverse = True):  
  
        # Function for sorting a list of tuples w.r.t a perticular element
        # k : index of the item in the list w.r.t which sorting is done
        # reverse = False : ascending
        # reverse = True : descending
        
        tup.sort(key = lambda x: x[k], reverse = reverse) 
        return tup  
    
    harris_sorted = Sort_Tuple(harris_tuple, 0, reverse = False)
    
    
    
    
    def ANMS(harris_sorted):
        # Function for Adapted Non Maximal Supression 
        
        def comp_dist(a, b):
            # Function for comuting the Eucledian Distance between th co-ordinates
            # .. of two featue points.
            
            a_x = harris_sorted[a][1]
            a_y = harris_sorted[a][2]
            
            b_x = harris_sorted[b][1]
            b_y = harris_sorted[b][2]
            
            output = np.sqrt((a_x-b_x)**2 + (a_y-b_y)**2)
            
            return output
        
        for i in range(0, len(harris_sorted)):
            
            dist = []
            
            for j in range(i+1, len(harris_sorted)):
                
                if harris_sorted[i][0] < 0.9*harris_sorted[j][0]:    # Robustness
                 
                    dist.append([j,comp_dist(i, j)])
            
            dist = Sort_Tuple(dist, 1, reverse = False)

            try :
                harris_sorted[i].append(dist[0][1])
            except IndexError:
                harris_sorted[i].append(100000000) 
                
            # we insert an arbitary large value for if the feature point doesn't
            # ..satisfy the robustness criterion
        
        return harris_sorted
    
    adapted = ANMS(harris_sorted)
    
    # The list adapted_sorted stores the tuple sorted w.r.t the ANMS radii
    # adapted_sorted = [response strength, x_co-ordinate, y_co-ordinate, ANMS radii]
    
    adapted_sorted = Sort_Tuple(adapted, 3, reverse = True)

    print("Total no. of feature points detected with ANMS: ", len(adapted_sorted))
    
    out_x = []
    out_y = []
    
    for i in range(0,len(adapted_sorted)):
        
        out_x.append(adapted_sorted[i][2])
        out_y.append(adapted_sorted[i][1])
        
    x_array = np.array(out_x)
    y_array = np.array(out_y)
    
    return x_array, y_array
