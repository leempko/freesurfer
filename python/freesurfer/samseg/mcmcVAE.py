import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import os
import numpy as np
import freesurfer.gems as gems
import copy
from freesurfer.samseg.utilities import requireNumpyArray
from scipy.ndimage.interpolation import zoom
from scipy.stats import invwishart
eps = np.finfo(float).eps


def getGaussianLikelihoods(data, mean, variance):
    #
    numberOfContrasts = data.shape[1]

    L = np.linalg.cholesky(variance)
    tmp = np.linalg.solve(L, data.T - mean)
    squaredMahalanobisDistances = np.sum(tmp ** 2, axis=0)
    sqrtDeterminantOfVariance = np.prod(np.diag(L))
    scaling = 1.0 / (2 * np.pi) ** (numberOfContrasts / 2) / sqrtDeterminantOfVariance
    gaussianLikelihoods = np.exp(squaredMahalanobisDistances * -0.5) * scaling
    return gaussianLikelihoods.T


def writeImage(fileName, buffer, cropping, example):
    # Write un-cropped image to file
    uncroppedBuffer = np.zeros(example.getImageBuffer().shape, dtype=np.float32, order='F')
    uncroppedBuffer[cropping] = buffer
    gems.KvlImage(requireNumpyArray(uncroppedBuffer)).write(fileName, example.transform_matrix)


# Encoder network
def get_encoder(data):
    epsilon = 1e-3

    graph = tf.get_default_graph()

    # First convolution
    conv_1_bias = graph.get_tensor_by_name('aconv_1/bias:0')
    conv_1_kernel = graph.get_tensor_by_name('aconv_1/kernel:0')
    conv_1 = tf.nn.conv3d(data, conv_1_kernel, strides=[1, 2, 2, 2, 1], padding='VALID') + conv_1_bias
    conv_1 = tf.nn.relu(conv_1)
    mu_1 = graph.get_tensor_by_name('abatch_n_e_1/moving_mean:0')
    var_1 = graph.get_tensor_by_name('abatch_n_e_1/moving_variance:0')
    beta_1 = graph.get_tensor_by_name('abatch_n_e_1/beta:0')
    gamma_1 = graph.get_tensor_by_name('abatch_n_e_1/gamma:0')
    batch_n_e_1 = gamma_1 * ((conv_1 - mu_1) / tf.sqrt(var_1 + epsilon)) + beta_1

    # Second convolution
    conv_2_bias = graph.get_tensor_by_name('aconv_2/bias:0')
    conv_2_kernel = graph.get_tensor_by_name('aconv_2/kernel:0')
    conv_2 = tf.nn.conv3d(batch_n_e_1, conv_2_kernel, strides=[1, 2, 2, 2, 1], padding='VALID') + conv_2_bias
    conv_2 = tf.nn.relu(conv_2)
    mu_2 = graph.get_tensor_by_name('abatch_n_e_2/moving_mean:0')
    var_2 = graph.get_tensor_by_name('abatch_n_e_2/moving_variance:0')
    beta_2 = graph.get_tensor_by_name('abatch_n_e_2/beta:0')
    gamma_2 = graph.get_tensor_by_name('abatch_n_e_2/gamma:0')
    batch_n_e_2 = gamma_2 * ((conv_2 - mu_2) / tf.sqrt(var_2 + epsilon)) + beta_2

    # Third convolution
    conv_3_bias = graph.get_tensor_by_name('aconv_3/bias:0')
    conv_3_kernel = graph.get_tensor_by_name('aconv_3/kernel:0')
    conv_3 = tf.nn.conv3d(batch_n_e_2, conv_3_kernel, strides=[1, 2, 2, 2, 1], padding='VALID') + conv_3_bias
    conv_3 = tf.nn.relu(conv_3)
    mu_3 = graph.get_tensor_by_name('abatch_n_e_3/moving_mean:0')
    var_3 = graph.get_tensor_by_name('abatch_n_e_3/moving_variance:0')
    beta_3 = graph.get_tensor_by_name('abatch_n_e_3/beta:0')
    gamma_3 = graph.get_tensor_by_name('abatch_n_e_3/gamma:0')
    batch_n_e_3 = gamma_3 * ((conv_3 - mu_3) / tf.sqrt(var_3 + epsilon)) + beta_3

    # Fourth convolution
    conv_4_bias = graph.get_tensor_by_name('aconv_4/bias:0')
    conv_4_kernel = graph.get_tensor_by_name('aconv_4/kernel:0')
    conv_4 = tf.nn.conv3d(batch_n_e_3, conv_4_kernel, strides=[1, 2, 2, 2, 1], padding='VALID') + conv_4_bias
    var_4 = graph.get_tensor_by_name('abatch_n_e_4/moving_variance:0')
    beta_4 = graph.get_tensor_by_name('abatch_n_e_4/beta:0')
    gamma_4 = graph.get_tensor_by_name('abatch_n_e_4/gamma:0')
    conv_4 = tf.nn.relu(conv_4)
    mu_4 = graph.get_tensor_by_name('abatch_n_e_4/moving_mean:0')
    batch_n_e_4 = gamma_4 * ((conv_4 - mu_4) / tf.sqrt(var_4 + epsilon)) + beta_4

    # Fifth convolution
    conv_5_bias = graph.get_tensor_by_name('aconv_5/bias:0')
    conv_5_kernel = graph.get_tensor_by_name('aconv_5/kernel:0')
    conv_5 = tf.nn.conv3d(batch_n_e_4, conv_5_kernel, strides=[1, 1, 1, 1, 1], padding='VALID') + conv_5_bias
    conv_5 = tf.nn.relu(conv_5)
    mu_5 = graph.get_tensor_by_name('abatch_n_e_5/moving_mean:0')
    var_5 = graph.get_tensor_by_name('abatch_n_e_5/moving_variance:0')
    beta_5 = graph.get_tensor_by_name('abatch_n_e_5/beta:0')
    gamma_5 = graph.get_tensor_by_name('abatch_n_e_5/gamma:0')
    batch_n_e_5 = gamma_5 * ((conv_5 - mu_5) / tf.sqrt(var_5 + epsilon)) + beta_5

    # Convolve to mu e sigma instead of using fully connected
    mu_bias = graph.get_tensor_by_name('amu/bias:0')
    mu_kernel = graph.get_tensor_by_name('amu/kernel:0')
    mu = tf.nn.conv3d(batch_n_e_5, mu_kernel, strides=[1, 1, 1, 1, 1], padding='VALID') + mu_bias

    sigma_bias = graph.get_tensor_by_name('asigma/bias:0')
    sigma_kernel = graph.get_tensor_by_name('asigma/kernel:0')
    sigma = tf.nn.conv3d(batch_n_e_5, sigma_kernel, strides=[1, 1, 1, 1, 1], padding='VALID') + sigma_bias
    sigma = tf.nn.softplus(sigma)

    return tfd.MultivariateNormalDiag(mu, sigma)


# decoder network
def get_decoder(code, imageSize):
    epsilon = 1e-3
    
    graph = tf.get_default_graph()
    
    code_size = tf.shape(code)

    # First deconv layer
    deconv_0_bias = graph.get_tensor_by_name('adeconv_0/bias:0')
    deconv_0_kernel = graph.get_tensor_by_name('adeconv_0/kernel:0')
    # Deconvolution shape for VALID = stride * (input - 1) + kernel size
    deconv_shape = tf.stack([code_size[0], code_size[1] - 1 + 3, code_size[2] - 1 + 3, code_size[3] - 1 + 3, 16])

    hidden_0_dec = tf.nn.conv3d_transpose(code, deconv_0_kernel, output_shape=deconv_shape,
                                          strides=[1, 1, 1, 1, 1],
                                          padding='VALID', ) + deconv_0_bias

    mu_1 = graph.get_tensor_by_name('abatch_n_d_0/moving_mean:0')
    var_1 = graph.get_tensor_by_name('abatch_n_d_0/moving_variance:0')
    beta_1 = graph.get_tensor_by_name('abatch_n_d_0/beta:0')
    gamma_1 = graph.get_tensor_by_name('abatch_n_d_0/gamma:0')
    batch_n_d_1 = gamma_1 * ((hidden_0_dec - mu_1) / tf.sqrt(var_1 + epsilon)) + beta_1

    # Second deconv layer
    code_size = tf.shape(batch_n_d_1)
    deconv_1_bias = graph.get_tensor_by_name('adeconv_1/bias:0')
    deconv_1_kernel = graph.get_tensor_by_name('adeconv_1/kernel:0')
    deconv_shape = tf.stack([code_size[0], code_size[1] - 1 + 3, code_size[2] - 1 + 3, code_size[3] - 1 + 3, 16])

    hidden_2_dec = tf.nn.conv3d_transpose(batch_n_d_1, deconv_1_kernel, output_shape=deconv_shape,
                                          strides=[1, 1, 1, 1, 1],
                                          padding='VALID') + deconv_1_bias

    hidden_2_dec = tf.nn.relu(hidden_2_dec)
    mu_2 = graph.get_tensor_by_name('abatch_n_d_1/moving_mean:0')
    var_2 = graph.get_tensor_by_name('abatch_n_d_1/moving_variance:0')
    beta_2 = graph.get_tensor_by_name('abatch_n_d_1/beta:0')
    gamma_2 = graph.get_tensor_by_name('abatch_n_d_1/gamma:0')
    batch_n_d_2 = gamma_2 * ((hidden_2_dec - mu_2) / tf.sqrt(var_2 + epsilon)) + beta_2

    # Third deconv layer
    code_size = tf.shape(batch_n_d_2)
    deconv_2_bias = graph.get_tensor_by_name('adeconv_2/bias:0')
    deconv_2_kernel = graph.get_tensor_by_name('adeconv_2/kernel:0')

    deconv_shape = tf.stack(
        [code_size[0], 2 * (code_size[1] - 1) + 5, 2 * (code_size[2] - 1) + 5, 2 * (code_size[3] - 1) + 5, 16])

    hidden_3_dec = tf.nn.conv3d_transpose(batch_n_d_2, deconv_2_kernel, output_shape=deconv_shape,
                                          strides=[1, 2, 2, 2, 1],
                                          padding='VALID') + deconv_2_bias

    hidden_3_dec = tf.nn.relu(hidden_3_dec)
    mu_3 = graph.get_tensor_by_name('abatch_n_d_2/moving_mean:0')
    var_3 = graph.get_tensor_by_name('abatch_n_d_2/moving_variance:0')
    beta_3 = graph.get_tensor_by_name('abatch_n_d_2/beta:0')
    gamma_3 = graph.get_tensor_by_name('abatch_n_d_2/gamma:0')
    batch_n_d_3 = gamma_3 * ((hidden_3_dec - mu_3) / tf.sqrt(var_3 + epsilon)) + beta_3

    # Fourth deconv layer
    code_size = tf.shape(batch_n_d_3)
    deconv_3_bias = graph.get_tensor_by_name('adeconv_3/bias:0')
    deconv_3_kernel = graph.get_tensor_by_name('adeconv_3/kernel:0')

    deconv_shape = tf.stack(
        [code_size[0], 2 * (code_size[1] - 1) + 5, 2 * (code_size[2] - 1) + 5, 2 * (code_size[3] - 1) + 5, 24])

    hidden_4_dec = tf.nn.conv3d_transpose(batch_n_d_3, deconv_3_kernel, output_shape=deconv_shape,
                                          strides=[1, 2, 2, 2, 1],
                                          padding='VALID') + deconv_3_bias

    hidden_4_dec = tf.nn.relu(hidden_4_dec)
    mu_4 = graph.get_tensor_by_name('abatch_n_d_3/moving_mean:0')
    var_4 = graph.get_tensor_by_name('abatch_n_d_3/moving_variance:0')
    beta_4 = graph.get_tensor_by_name('abatch_n_d_3/beta:0')
    gamma_4 = graph.get_tensor_by_name('abatch_n_d_3/gamma:0')
    batch_n_d_4 = gamma_4 * ((hidden_4_dec - mu_4) / tf.sqrt(var_4 + epsilon)) + beta_4

    # Fifth deconv layer
    code_size = tf.shape(batch_n_d_4)
    deconv_4_bias = graph.get_tensor_by_name('adeconv_4/bias:0')
    deconv_4_kernel = graph.get_tensor_by_name('adeconv_4/kernel:0')

    deconv_shape = tf.stack(
        [code_size[0], 2 * (code_size[1] - 1) + 5, 2 * (code_size[2] - 1) + 5, 2 * (code_size[3] - 1) + 5, 32])

    hidden_5_dec = tf.nn.conv3d_transpose(batch_n_d_4, deconv_4_kernel, output_shape=deconv_shape,
                                          strides=[1, 2, 2, 2, 1],
                                          padding='VALID') + deconv_4_bias

    hidden_5_dec = tf.nn.relu(hidden_5_dec)
    mu_5 = graph.get_tensor_by_name('abatch_n_d_4/moving_mean:0')
    var_5 = graph.get_tensor_by_name('abatch_n_d_4/moving_variance:0')
    beta_5 = graph.get_tensor_by_name('abatch_n_d_4/beta:0')
    gamma_5 = graph.get_tensor_by_name('abatch_n_d_4/gamma:0')
    batch_n_d_5 = gamma_5 * ((hidden_5_dec - mu_5) / tf.sqrt(var_5 + epsilon)) + beta_5

    # Sixth deconv layer
    code_size = tf.shape(batch_n_d_5)
    deconv_5_bias = graph.get_tensor_by_name('adeconv_5/bias:0')
    deconv_5_kernel = graph.get_tensor_by_name('adeconv_5/kernel:0')
    deconv_shape = tf.stack(
        [code_size[0], 2 * (code_size[1] - 1) + 5, 2 * (code_size[2] - 1) + 5, 2 * (code_size[3] - 1) + 5, 1])

    hidden_6_dec = tf.nn.conv3d_transpose(batch_n_d_5, deconv_5_kernel, output_shape=deconv_shape,
                                          strides=[1, 2, 2, 2, 1],
                                          padding='VALID') + deconv_5_bias

    # We put -100 to the padding to be sure that the final prob is almost if not zero
    hidden_6_dec = pad_up_to(hidden_6_dec, [1, imageSize[0], imageSize[1], imageSize[2], 1], constant_values=-100)

    return tf.nn.sigmoid(hidden_6_dec)


# add paddings when the size last layer does not match the input size, this happens for deconvolution layers
def pad_up_to(t, max_in_dims, constant_values):
    s = tf.shape(t)
    paddings = [[tf.cast((m - s[i]) / 2, tf.int32), (m - s[i]) - tf.cast((m - s[i]) / 2, tf.int32)] for (i, m) in
                enumerate(max_in_dims)]
    return tf.pad(t, tf.cast(tf.stack(paddings), tf.int32), 'CONSTANT', constant_values=constant_values)


# here is the net function. The input goes through the encoder, we sample from it and then it goes through the decoder
def run_net(lesion, imageSize):
    code = get_encoder(lesion).sample()
    return get_decoder(code, imageSize)


# pad function for scaling to image size from net size
def scale(data, imageSize, net_size):
    diff_1 = ((net_size - imageSize) / 2).astype(int)
    diff_2 = ((net_size - imageSize) - diff_1).astype(int)
    if diff_2[0] < 0:
        diff_1[0] = 0
        diff_2[0] = 0
    if diff_2[1] < 0:
        diff_1[1] = 0
        diff_2[1] = 0
    if diff_2[2] < 0:
        diff_1[2] = 0
        diff_2[2] = 0
    if len(data.shape) == 3:
        paddings = [[diff_1[0], diff_2[0]], [diff_1[1], diff_2[1]], [diff_1[2], diff_2[2]]]
    else:
        paddings = [[0, 0], [diff_1[0], diff_2[0]], [diff_1[1], diff_2[1]], [diff_1[2], diff_2[2]], [0, 0]]

    return np.pad(data, paddings, mode='constant')


# pad function for cropping to image size from net size
def crop(data, imageSize, net_size):
    diff_1 = ((net_size - imageSize) / 2).astype(int)
    diff_2 = ((net_size - imageSize) - diff_1).astype(int)
    if diff_2[0] < 0:
        diff_1[0] = 0
        diff_2[0] = 0
    if diff_2[1] < 0:
        diff_1[1] = 0
        diff_2[1] = 0
    if diff_2[2] < 0:
        diff_1[2] = 0
        diff_2[2] = 0
    if len(data.shape) == 3:
        return data[diff_1[0]:(imageSize[0] + diff_1[0]), diff_1[1]:(imageSize[1] + diff_1[1]),
               diff_1[2]:(imageSize[2] + diff_1[2])]
    data = data[0, diff_1[0]:(imageSize[0] + diff_1[0]), diff_1[1]:(imageSize[1] + diff_1[1]),
           diff_1[2]:(imageSize[2] + diff_1[2])]
    return data


# Sample function
def sampleVAE(
        lesionOptions,
        imageSize,
        modelPath,
        maskIndices,
        atlas,
        lesion_idx,
        dataMask,
        lesionsInit,
        savePath,
        voxelSpacing,
        FreeSurferLabels,
        gm_idx,
        cropping,
        imageFileNames,
        names,
        means,
        variances,
        mixtureWeights,
        hyperMean,
        hyperVariance,
        hyperMeanNumberOfMeasurements,
        hyperVarianceNumberOfMeasurements,
        numberOfGaussiansPerClass,
        gaussianNumber_lesion,
        posteriors,
        fractionsTable,
        useDiagonalCovarianceMatrices
):

    # Restore VAE model from checkpoint
    sess = tf.Session()
    saver = tf.train.import_meta_graph(modelPath + '/model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint(modelPath))
    print('VAE lesion model loaded')

    # First compute likelihood from initial parameter estimation
    numberOfClasses = len(numberOfGaussiansPerClass)
    numberOfVoxels = dataMask.shape[0]
    numberOfStructures = fractionsTable.shape[1]
    likelihoods = np.zeros_like(posteriors, dtype=np.float64)
    for classNumber in range(numberOfClasses):

        # Compute likelihood for this class
        classLikelihoods = np.zeros(numberOfVoxels)
        numberOfComponents = numberOfGaussiansPerClass[classNumber]
        for componentNumber in range(numberOfComponents):
            gaussianNumber = sum(numberOfGaussiansPerClass[:classNumber]) + componentNumber
            mean = np.expand_dims(means[gaussianNumber, :], 1)
            variance = variances[gaussianNumber, :, :]
            mixtureWeight = mixtureWeights[gaussianNumber]

            gaussianLikelihoods = getGaussianLikelihoods(dataMask, mean, variance)
            classLikelihoods += gaussianLikelihoods * mixtureWeight

        # Add contribution to the actual structures
        for structureNumber in range(numberOfStructures):
            fraction = fractionsTable[classNumber, structureNumber]
            if fraction < 1e-10:
                continue
            likelihoods[:, structureNumber] += classLikelihoods * fraction

    # Create lesion and outlier masks
    # For mask if we have -1 mask below WM mean, +1 above, 0 nothing
    tmp = np.zeros(imageSize, np.uint8)
    tmp[lesionsInit] = 1
    mask = np.ones(imageSize, dtype=bool)
    k = 0
    gm_means = means[gm_idx, :]
    for i in lesionOptions['lesionMask']:
        data = np.zeros(imageSize[0] * imageSize[1] * imageSize[2])
        data[np.reshape(maskIndices, [-1]) == 1] = dataMask[:, k]
        data = np.reshape(data, imageSize)
        if i == '-1':
            tmp = data < gm_means[k]
        elif i == '1':
            tmp = data > gm_means[k]
        elif i == '0':
            tmp = np.ones(imageSize, dtype=bool)
        mask = np.logical_and(mask, tmp)
        k = k + 1

    # Center image
    coords = np.argwhere(data > 0)
    x0, y0, z0 = coords.min(axis=0)
    x1, y1, z1 = coords.max(axis=0) + 1  # slices are exclusive at the top
    mask = mask[x0:x1, y0:y1, z0:z1]

    # Get voxel resolution
    dilations = [1 / voxelSpacing[0], 1 / voxelSpacing[1], 1 / voxelSpacing[2]]

    # Size of the training image for the VAE
    net_size = np.array([197, 233, 189])

    # Paddings for go back to original image size
    paddings = [[x0, imageSize[0] - x1], [y0, imageSize[1] - y1], [z0, imageSize[2] - z1]]

    # Create lesion input
    tmp = np.zeros(imageSize)
    tmp[lesionsInit] = 1
    tmp = tmp[x0:x1, y0:y1, z0:z1]

    lesions = tmp * mask
    
    # Zoom it 1x1x1 mm resolution
    lesions = zoom(lesions, voxelSpacing, order=1)

    net_shape = np.array([197, 233, 189])
    # If input image is smaller than net_shape we need to pad
    if net_shape[0] > lesions.shape[0] or net_shape[1] > lesions.shape[1] or net_shape[2] > lesions.shape[2]:
        pad = True
        if net_shape[0] < lesions.shape[0]:
            net_shape[0] = lesions.shape[0]
        if net_shape[1] < lesions.shape[1]:
            net_shape[1] = lesions.shape[1]
        if net_shape[2] < lesions.shape[2]:
            net_shape[2] = lesions.shape[2]
    else:
        net_shape = np.array([imageSize[0], imageSize[1], imageSize[2]])
        pad = False
    imageSize = np.array([imageSize[0], imageSize[1], imageSize[2]])

    # Create tf placeholder
    lesionPlaceholder = tf.placeholder(tf.float32, [1, net_shape[0], net_shape[1], net_shape[2], 1])
    net = run_net(lesionPlaceholder, net_shape)
    lesions = np.expand_dims(np.expand_dims(lesions, 0), 4)

    if pad:
        lesion = scale(lesions, [lesions.shape[1], lesions.shape[2], lesions.shape[3]], net_size)
    else:
        lesion = lesions

    posteriors_collected = np.zeros_like(posteriors)

    for i in range(lesionOptions['samplingSteps'] + lesionOptions['burnInSteps']):
        # Sample mean and variance for lesion gaussian
        # First we sample from the means given the variances
        # then we sample from the variances given the means
        # we then recompute the likelihood for the lesion gaussian
        for t in range(numberOfClasses):
            numberOfComponents = numberOfGaussiansPerClass[t]
            for componentNumber in range(numberOfComponents):
                gaussianNumber = int(np.sum(numberOfGaussiansPerClass[: t]) + componentNumber)
                if gaussianNumber == gaussianNumber_lesion:
                    posterior = posteriors[:, lesion_idx]
                    posterior = posterior.reshape(-1, 1)

                    mean = (dataMask.T @ posterior + hyperMean.T * hyperMeanNumberOfMeasurements) \
                           / (np.sum(posterior) + hyperMeanNumberOfMeasurements)
                    means[gaussianNumber, :] = np.random.multivariate_normal(mean.ravel(),
                                                                             variances[gaussianNumber, :, :] / (np.sum(
                                                                                 posterior) + hyperMeanNumberOfMeasurements))
                    tmp = dataMask - mean.T
                    S = tmp.T @ (tmp * posterior) + \
                        ((hyperMeanNumberOfMeasurements * np.sum(posterior)) / (np.sum(posterior) + hyperMeanNumberOfMeasurements)
                         * ((mean - hyperMean) @ (mean - hyperMean).T)) + \
                            hyperVariance * hyperVarianceNumberOfMeasurements
                    variances[gaussianNumber, :, :] = invwishart.rvs(
                        np.sum(posterior) + 1 + hyperVarianceNumberOfMeasurements, S)
                    if useDiagonalCovarianceMatrices:
                        # Force diagonal covariance matrices
                        variances[gaussianNumber, :, :] = np.diag(np.diag(variances[gaussianNumber, :, :]))
                    mean = np.expand_dims(means[gaussianNumber, :], 1)
                    variance = variances[gaussianNumber, :, :]

                    likelihoods[:, lesion_idx] = getGaussianLikelihoods( dataMask, mean, variance )

        # Run network and get prior
        if pad:
            data = crop(np.squeeze(np.squeeze(sess.run(net, {lesionPlaceholder: lesion}), 4), 0),
                        [lesions.shape[1], lesions.shape[2], lesions.shape[3]], net_size)
            data = np.clip(zoom(data, dilations), 0, 1)
            # If we have some rounding problems from zoom, pad with 0 for the rounding errors.
            if data.shape < mask.shape:
                data = scale(data, np.array([data.shape[0], data.shape[1], data.shape[2]]),
                             np.array([mask.shape[0], mask.shape[1], mask.shape[2]]))
            if data.shape > mask.shape:
                data = crop(data, np.array([mask.shape[0], mask.shape[1], mask.shape[2]]),
                            np.array([data.shape[0], data.shape[1], data.shape[2]]))
        else:
            data = np.squeeze(np.squeeze(sess.run(net, {lesionPlaceholder: lesion}), 4), 0)
            data = np.clip(zoom(data, dilations), 0, 1)

        # Assign lesion prior
        priors = np.array(atlas / 65535, dtype=np.float32)
        p_net_lesion = np.reshape(np.pad((data * mask), paddings, mode='constant')[maskIndices == 1], -1).astype(
            np.float32)
        priors[:, lesion_idx] = p_net_lesion * priors[:, lesion_idx]
        normalizer = np.sum(priors, axis=1) + eps
        priors = priors / np.expand_dims(normalizer, 1)

        posteriors = priors * likelihoods

        # Normalize posteriors
        normalizer = np.sum(posteriors, axis=1) + eps
        posteriors = posteriors / np.expand_dims(normalizer, 1)

        # Multinomial sampling for posteriors in order to get new lesion sample
        print('Multinomial sampling')
        diff = np.cumsum(posteriors, axis=1) - np.random.random([posteriors.shape[0], 1])
        diff[diff < 0] = 1.1
        diff = np.argmin(diff, axis=1)
        lesion_s = np.zeros(imageSize[0] * imageSize[1] * imageSize[2])
        lesion_s[np.reshape(maskIndices == 1, -1)] = (diff == lesion_idx)
        lesion_s = np.reshape(lesion_s, imageSize)
        lesion_s = lesion_s[x0:x1, y0:y1, z0:z1]

        # Prepare data to feed to the network for next loop cycle
        lesion_s = zoom(lesion_s, voxelSpacing, order=1)
        lesion_s = np.expand_dims(np.expand_dims(lesion_s, 0), 4)
        if pad:
            lesion = scale(lesion_s, [lesions.shape[1], lesions.shape[2], lesions.shape[3]], net_size)
        else:
            lesion = lesion_s

        # Collect data after burn in steps
        if i >= lesionOptions['burnInSteps']:
            print('Sample ' + str(i + 1 - lesionOptions['burnInSteps']) + ' times')
            posteriors_collected = posteriors_collected + posteriors
        else:
            print('Burn-in ' + str(i + 1) + ' times')

    # Average samples and save images
    posteriors = posteriors_collected / float(lesionOptions['samplingSteps'])
    # Save lesion posteriors without thresholding for writing images later on
    posteriorsLes = copy.deepcopy(posteriors[:, lesion_idx])
    # Threshold lesions. If we put 0 and 1 we don't need to care about the other prob structure when taking the maximum
    posteriors[:, lesion_idx][posteriors[:, lesion_idx] > lesionOptions['threshold']] = 1
    posteriors[:, lesion_idx][posteriors[:, lesion_idx] < lesionOptions['threshold']] = 0

    # Write the segmentation and the posteriors out
    structureNumbers = np.array(np.argmax(posteriors, 1), dtype=np.uint32)
    labels = np.zeros(imageSize, dtype=np.uint16)
    FreeSurferLabels = np.array(FreeSurferLabels, dtype=np.uint16)
    labels[maskIndices == 1] = FreeSurferLabels[structureNumbers]

    # Write out various images - segmentation first
    exampleImage = gems.KvlImage( imageFileNames[ 0 ] )
    image_base_path, _ = os.path.splitext( imageFileNames[ 0 ] )
    _, scanName = os.path.split( image_base_path )
    writeImage( os.path.join( savePath, scanName + '_FinalcrispSegmentation.nii' ), labels, cropping, exampleImage )

    # Also write the posteriors if flag on
    if lesionOptions['savePosteriors']:
        for classNumber in range(atlas.shape[1]):
            labels = np.zeros(imageSize, dtype=np.float32)
            if classNumber != lesion_idx:
                labels[maskIndices == 1] = posteriors[:, classNumber]
                writeImage(os.path.join(savePath, names[classNumber] + '_posterior.nii'), labels, cropping, exampleImage)
            else:
                labels[maskIndices == 1] = posteriorsLes
                writeImage(os.path.join(savePath, names[classNumber] + '_posterior.nii'), labels, cropping, exampleImage)

    # Compute volumes in mm^3
    volumeOfOneVoxel = np.abs( np.linalg.det( exampleImage.transform_matrix.as_numpy_array[ 0:3, 0:3 ] ) )
    volumesInCubicMm = ( np.sum( posteriors, axis=0 ) ) * volumeOfOneVoxel

    return [FreeSurferLabels, volumesInCubicMm]

