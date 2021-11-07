"""


COLOR TRANSFER FILE


"""
import numpy as np


def colorTransfer(src, tgt):
    src_img = np.copy(src)
    tgt_img = np.copy(tgt)

    src_lab = np.copy(src)
    tgt_lab = np.copy(tgt)

    """

    STEP 1

    """
    cone_matrix = np.array([
        [0.3811, 0.5783, 0.0402],
        [0.1967, 0.7244, 0.0782],
        [0.0241, 0.1288, 0.8444]])

    src_lms = np.copy(src_img)
    tgt_lms = np.copy(tgt_img)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src_lms[i,j] = np.matmul(cone_matrix, src_img[i,j])

    for i in range(tgt.shape[0]):
        for j in range(tgt.shape[1]):
            tgt_lms[i,j] = np.matmul(cone_matrix, tgt_img[i,j])


    """

    STEP 2

    """
    # convert lms matrices to the logaritmic space

    #src_log_lms = np.where(src_lms == 0, src_lms, np.log(src_lms))
    #tgt_log_lms = np.where(tgt_lms == 0, tgt_lms, np.log(tgt_lms))
    src_log_lms = np.log(src_lms)
    tgt_log_lms = np.log(tgt_lms)


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

    STEP 5
    
    """

    # Subtract the mean of source image from the source image
    src_lab[:,:,0] = src_lab[:,:,0] - mean_l_src
    src_lab[:,:,1] = src_lab[:,:,1] - mean_a_src
    src_lab[:,:,2] = src_lab[:,:,2] - mean_b_src

    """

    STEP 6
    
    """

    src_lab[:,:,0] = (var_l_tgt/var_l_src)*src_lab[:,:,0]
    src_lab[:,:,1] = (var_a_tgt/var_a_src)*src_lab[:,:,1]
    src_lab[:,:,2] = (var_b_tgt/var_b_src)*src_lab[:,:,2]


    """

    STEP 7
    
    """

    src_lab[:,:,0] = src_lab[:,:,0] + mean_l_tgt
    src_lab[:,:,1] = src_lab[:,:,1] + mean_a_tgt
    src_lab[:,:,2] = src_lab[:,:,2] + mean_b_tgt


    """

    STEP 8
    
    """
    
    lab_mat3 = np.array([[1,1,1], [1,1,-1],[1,-2,0]])
    lab_mat4 = np.array([
        [3/np.sqrt(3), 0, 0], 
        [0, np.sqrt(6)/6, 0],
        [0,0,2/np.sqrt(2)]])

    lab_mat_res = np.matmul(lab_mat3,lab_mat4)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            src_lab[i,j] = np.matmul(lab_mat_res, src_lab[i,j])
    
    """

    STEP 9
    
    """
    src_lab = src_lab**10


    """

    STEP 10
    
    """

    new_mat = np.array([
        [4.4679,-3.5873, 0.1193],
        [-1.2186,2.3809,-0.1624],
        [0.0497, -0.2439, 1.2045]])

    result_img = np.copy(src)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            result_img[i,j] = np.matmul(new_mat, src_lab[i,j])

    return result_img
