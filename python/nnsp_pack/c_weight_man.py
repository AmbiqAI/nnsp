"""
C code weight table management
"""
import numpy as np

def c_matrix_man(mat):
    """
    C code matrix weight table management
    """
    rows, cols = mat.shape
    mat_reshape = np.array([])
    blks_r = rows >> 2
    residue_r = rows % 4
    blks_c = cols >> 1
    residue_c = cols % 2

    for i in range(blks_r):
        blk = mat[i*4 : i*4+4,:]
        for val in range(blks_c):
            sub_blk = blk[:, val*2 : val*2+2].flatten()
            mat_reshape = np.concatenate([mat_reshape, sub_blk])
        if residue_c:
            sub_blk = blk[:,-1].flatten()
            mat_reshape = np.concatenate((mat_reshape, sub_blk))

    if residue_r:
        blk = mat[-residue_r:,:]
        for val in range(blks_c):
            sub_blk = blk[:, val*2:val*2+2].flatten()
            mat_reshape = np.concatenate((mat_reshape, sub_blk))
        if residue_c:
            sub_blk = blk[:,-1].flatten()
            mat_reshape = np.concatenate((mat_reshape, sub_blk))

    return mat_reshape

def c_lstm_weight_man(kernel_fw, kernel_re, bias):
    """
    C code lstm weight table management
    """
    rows = kernel_fw.shape[0] >> 2
    kernel_fw_reshape  = c_lstm_kernel_man(kernel_fw, rows)
    kernel_re_reshape = c_lstm_kernel_man(kernel_re, rows)

    bias_reshape  = c_lstm_bias_man(bias, rows)

    return kernel_fw_reshape, kernel_re_reshape, bias_reshape

def c_lstm_kernel_man(kernel, rows):
    """
    C code lstm kernel table management
    """
    i_kernel_tot, j_kernel_tot, f_kernel_tot, o_kernel_tot = np.split(kernel, 4, axis=0) # pylint: disable=unbalanced-tuple-unpacking
    blks_r = rows >> 2
    residue_r = rows % 4

    kernel_reshape = np.array([])

    for i in range(blks_r):
        i_kernel = i_kernel_tot[i*4:i*4+4,:]
        j_kernel = j_kernel_tot[i*4:i*4+4,:]
        f_kernel = f_kernel_tot[i*4:i*4+4,:]
        o_kernel = o_kernel_tot[i*4:i*4+4,:]

        kernel_reshape = np.concatenate((kernel_reshape, c_matrix_man(i_kernel)))
        kernel_reshape = np.concatenate((kernel_reshape, c_matrix_man(j_kernel)))
        kernel_reshape = np.concatenate((kernel_reshape, c_matrix_man(f_kernel)))
        kernel_reshape = np.concatenate((kernel_reshape, c_matrix_man(o_kernel)))

    if residue_r:
        i_kernel = i_kernel_tot[-residue_r:,:]
        j_kernel = j_kernel_tot[-residue_r:,:]
        f_kernel = f_kernel_tot[-residue_r:,:]
        o_kernel = o_kernel_tot[-residue_r:,:]
        kernel_reshape = np.concatenate((kernel_reshape, c_matrix_man(i_kernel)))
        kernel_reshape = np.concatenate((kernel_reshape, c_matrix_man(j_kernel)))
        kernel_reshape = np.concatenate((kernel_reshape, c_matrix_man(f_kernel)))
        kernel_reshape = np.concatenate((kernel_reshape, c_matrix_man(o_kernel)))

    return kernel_reshape

def c_lstm_bias_man(bias, rows):
    """
    C code lstm bias table management
    """
    i_bias_tot, j_bias_tot, f_bias_tot, o_bias_tot = np.split(bias, 4, axis=0) # pylint: disable=unbalanced-tuple-unpacking
    gps = rows >> 2
    rem = rows % 4
    bias_reshape = np.array([])

    for i in range(gps):
        i_bias = i_bias_tot[i*4:i*4+4]
        j_bias = j_bias_tot[i*4:i*4+4]
        f_bias = f_bias_tot[i*4:i*4+4]
        o_bias = o_bias_tot[i*4:i*4+4]

        bias_reshape = np.concatenate((bias_reshape, i_bias.flatten()))
        bias_reshape = np.concatenate((bias_reshape, j_bias.flatten()))
        bias_reshape = np.concatenate((bias_reshape, f_bias.flatten()))
        bias_reshape = np.concatenate((bias_reshape, o_bias.flatten()))
    if rem:
        i_bias = i_bias_tot[-rem:]
        j_bias = j_bias_tot[-rem:]
        f_bias = f_bias_tot[-rem:]
        o_bias = o_bias_tot[-rem:]

        bias_reshape = np.concatenate((bias_reshape, i_bias.flatten()))
        bias_reshape = np.concatenate((bias_reshape, j_bias.flatten()))
        bias_reshape = np.concatenate((bias_reshape, f_bias.flatten()))
        bias_reshape = np.concatenate((bias_reshape, o_bias.flatten()))

    return bias_reshape
