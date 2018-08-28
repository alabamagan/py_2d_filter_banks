from FilterBanks import DirectionalFilterBankDown, DirectionalFilterBankUp
from imageio import imread
from numpy.fft import fftshift, fft2, ifft2, ifftshift

import matplotlib.pyplot as plt
import numpy as np

from Utility import display_images

''' Testing'''
if __name__ == '__main__':

    '''Prepare input image'''
    im = imread('./Materials/s0.tif')
    imgt = imread('./Materials/gt.tif')

    '''Instruction to manual phase shift'''
    # # These phase shift operations can produce fan filter effect, but introduce the need to do fftshift
    # x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    # px = np.zeros(im.shape, dtype=np.complex)
    # px.imag = -np.pi*x
    # px = np.exp(px)
    # im = im*px

    # shift the input so that origin is at center of image
    s_fftim = fftshift(fft2(ifftshift(im)))
    s_sfftimgt = fftshift(fft2(ifftshift(imgt)))

    '''Create filter tree'''
    P_0 = DirectionalFilterBankDown()
    U_0 = DirectionalFilterBankUp()

    '''Exchange some of the subbands components'''
    s0_subbands = P_0.run(s_fftim)
    gt_subbands = P_0.run(s_sfftimgt)

    for i in xrange(0, 7):
        s0_subbands[:,:,i] = gt_subbands[:,:,i]

    recovered = U_0.run(s0_subbands)

    '''Display recovered image'''
    show = np.stack([ifftshift(ifft2(fftshift(recovered)).real),
                     np.abs(ifftshift(ifft2(fftshift(recovered)).real) - im),
                     np.abs(ifftshift(ifft2(fftshift(recovered)).real) - imgt),
                     im,
                     imgt], axis=-1)

    display_images(show, ncol=3)
