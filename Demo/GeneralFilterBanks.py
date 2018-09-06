from FilterBanks import Decimation, Interpolation
from imageio import imread
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
import numpy as np

from FilterBanks.Functions.Utility import display_subbands

if __name__ == '__main__':
    im = imread('../../Materials/lena_gray.png')[:,:,0]

    '''Introduce phase shift in image domain'''
    # # These phase shift operations can produce fan filter effect, but introduce the need to do fftshift
    # x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    # px = np.zeros(im.shape, dtype=np.complex)
    # px.imag = -np.pi*x
    # px = np.exp(px)
    # im = im*px

    # shift the input so that origin is at center of image
    s_fftim = fftshift(fft2(ifftshift(im)))

    '''
    Sampling matrix setting
    '''
    H_0 = Decimation()
    H_0.set_core_matrix(np.array([[2, 0], [1, 1]]))  # Manually set resampling matrix
    G_0 = Interpolation()
    G_0.set_core_matrix(np.array([[2, 0], [1, 1]]))  # Upsample and downsample should have same matrix
    G_0.hook_input(H_0)
    out = G_0.run(s_fftim)

    '''
    Display subband components
    '''
    display_subbands(out, ncol=3)

    '''Test actual recovered image'''
    plt.imshow(ifft2(fftshift(out)).real)           # Some times need fftshift, sometimes doesn't
    plt.show()

    '''Test frequency support calculation'''
    display_subbands(G_0._support)


