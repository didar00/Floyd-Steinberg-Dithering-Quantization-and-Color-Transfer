"""


COLOR TRANSFER FILE


"""
import numpy as np
from matplotlib import pyplot as plt


def colorTransfer(src, tgt):

    fig = plt.figure(figsize=(16, 16))
    fig.add_subplot(2,5,1)
    plt.imshow(src.astype(np.uint8))
    plt.title("source")
    
    fig.add_subplot(2,5,2)
    plt.imshow(tgt.astype(np.uint8))
    plt.title("target")


    src_lab = np.copy(src)
    tgt_lab = np.copy(tgt)

    src_lms = np.zeros(src.shape)
    tgt_lms = np.zeros(tgt.shape)
    
    """

    STEP 1

    """
    cone_matrix = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]])


    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src_lms[i,j] = np.matmul(cone_matrix, src_lab[i,j])


    for i in range(tgt.shape[0]):
        for j in range(tgt.shape[1]):
            tgt_lms[i,j] = np.matmul(cone_matrix, tgt_lab[i,j])

    fig.add_subplot(2,5,3)
    plt.imshow(_scale_array(src_lms, clip=False).astype(np.uint8))
    plt.title("step 1")
   
    

    """

    STEP 2

    """
    
    # convert lms matrices to the logaritmic space
  
    src_lms = _min_max_scale(src_lms, new_range=(1,255))
    tgt_lms = _min_max_scale(tgt_lms, new_range=(1,255))
    
    src_log_lms = np.log10(src_lms)
    tgt_log_lms = np.log10(tgt_lms)

    src_log_lms = np.where(src_log_lms == 1, 0, src_log_lms)
    tgt_log_lms = np.where(tgt_log_lms == 1, 0, tgt_log_lms)

    fig.add_subplot(2,5,4)
    plt.imshow(_min_max_scale(src_log_lms, new_range=(0,1)))
    plt.title("step 2")


    """

    STEP 3

    """

    lab_mat1 = np.array([
        [1/np.sqrt(3), 0, 0], 
        [0, 1/np.sqrt(6), 0],
        [0,0,1/np.sqrt(2)]])

    lab_mat2 = np.array([[1,1,1], [1,1,-2],[1,-1,0]])

    
    # calculate lab
    val_mat = np.matmul(lab_mat1, lab_mat2)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src_lab[i,j] = np.matmul(val_mat, src_log_lms[i,j])
 
    for i in range(tgt.shape[0]):
        for j in range(tgt.shape[1]):
            tgt_lab[i,j] = np.matmul(val_mat, tgt_log_lms[i,j])
     
    fig.add_subplot(2,5,5)
    plt.imshow(src_lab)
    plt.title("step 3")


    fig.add_subplot(2,5,6)
    plt.imshow(_min_max_scale(src_lab[:,:,0], new_range=(0,255)).astype(np.uint8))
    plt.title("L")
    fig.add_subplot(2,5,7)
    plt.imshow(_min_max_scale(src_lab[:,:,1], new_range=(0,255)).astype(np.uint8))
    plt.title("a")
    fig.add_subplot(2,5,8)
    plt.imshow(_min_max_scale(src_lab[:,:,2], new_range=(0,255)).astype(np.uint8))
    plt.title("b")



    """

    STEP 4
    
    """
    mean_l_src = src_lab[:,:,0].mean()
    mean_a_src = src_lab[:,:,1].mean()
    mean_b_src = src_lab[:,:,2].mean()


    var_l_src = src_lab[:,:,0].var()
    var_a_src = src_lab[:,:,1].var()
    var_b_src = src_lab[:,:,2].var()


    mean_l_tgt = tgt_lab[:,:,0].mean()
    mean_a_tgt = tgt_lab[:,:,1].mean()
    mean_b_tgt = tgt_lab[:,:,2].mean()

    var_l_tgt = tgt_lab[:,:,0].var()
    var_a_tgt = tgt_lab[:,:,1].var()
    var_b_tgt = tgt_lab[:,:,2].var()


    """
    
    COMBINE STEPS 5-7

    """

    
    src_lab[:,:,0] = (src_lab[:,:,0] - mean_l_src)*(var_l_tgt/var_l_src) + mean_l_tgt
    src_lab[:,:,1] = (src_lab[:,:,1] - mean_a_src)*(var_a_tgt/var_a_src) + mean_a_tgt
    src_lab[:,:,2] = (src_lab[:,:,2] - mean_b_src)*(var_b_tgt/var_b_src) + mean_b_tgt

   

    """

    STEP 8
    
    """
    
    lab_mat3 = np.array([[1,1,1], [1,1,-1],[1,-2,0]])
    lab_mat4 = np.array([
        [np.sqrt(3)/3, 0, 0],
        [0, np.sqrt(6)/6, 0],
        [0,0,np.sqrt(2)/2]])

    lab_mat_res = np.matmul(lab_mat3,lab_mat4)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src_lab[i,j] = np.matmul(lab_mat_res, src_lab[i,j])

    fig.add_subplot(2,5,9)
    plt.imshow(_min_max_scale(src_lab, new_range=(0,255)).astype(np.uint8))
    plt.title("step 8")


    """

    STEP 9
    
    """

    src_lab = 10**src_lab

    fig.add_subplot(2,5,10)
    plt.imshow(_min_max_scale(src_lab, new_range=(0,255)).astype(np.uint8))
    plt.title("step 9")
    plt.show()
    plt.clf()


    """

    STEP 10
    
    """

    new_mat = np.array([
        [4.4679,-3.5873, 0.1193],
        [-1.2186,2.3809,-0.1624],
        [0.0497, -0.2439, 1.2045]])

    result_img = np.zeros(src.shape)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            result_img[i,j] = np.matmul(new_mat, src_lab[i,j])

    result_img = _min_max_scale(result_img, new_range=(0,255))

    return result_img


def _min_max_scale(arr, new_range=(0, 255)):
	# get array's current min and max
	mn = arr.min()
	mx = arr.max()

	# check if scaling needs to be done to be in new_range
	if (mn < new_range[0] or mx > new_range[1]) and mn != mx:
		# perform min-max scaling
		scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
	else:
		# return array if already in range
		scaled = arr

	return scaled

def _scale_array(arr, clip=True):
	if clip:
		scaled = np.clip(arr, 0, 255)
	else:
		scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
		scaled = _min_max_scale(arr, new_range=scale_range)

	return scaled