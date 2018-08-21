from FilterBanks import Downsample, Upsample
from imageio import imread
from numpy.fft import fftshift, fft2, ifft2, ifftshift
import matplotlib.pyplot as plt
import numpy as np

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
    H_0 = Downsample()
    # H_1 = Downsample()
    # H_0.set_core_matrix(np.array([[2, 0], [0, 1]]))
    H_0.set_core_matrix(np.array([[2, 0], [1, 1]]))
    # H_0.set_core_matrix(np.array([[2, -2], [3, 2]]))
    G_0 = Upsample()
    # G_1 = Upsample()
    # G_0.set_core_matrix(np.array([[2, 0], [0, 1]]))
    G_0.set_core_matrix(np.array([[2, 0], [1, 1]]))
    # G_0.set_core_matrix(np.array([[2, -2], [3, 2]]))
    G_0.hook_input(H_0)
    # out = H_0.run(s_fftim)
    out = G_0.run(s_fftim)

    '''
    Display subband components
    '''
    M = H_0._outflow.shape[-1]
    ncol=2
    fig, axs = plt.subplots(M / ncol, ncol)
    for i in xrange(M):
        if M <= ncol:
            try:
                # axs[i].imshow(ifft2(fftshift(H_0._outflow[:,:,i])).real)
                axs[i].imshow(np.abs(H_0._outflow[:,:,i]), vmin=0, vmax=2500)
            except:
                pass
            # axs[i].imshow(ifft2(H_0._outflow[:,:,i]).real)
        else:
            # axs[i//ncol, i%ncol].imshow(ifft2(fftshift(H_0._outflow[:,:,i])).real)
            axs[i//ncol, i%ncol].imshow(np.abs(H_0._outflow[:,:,i]), vmin=0, vmax=2500)
    plt.show()

    '''Test actual recovered image'''
    # plt.scatter(out[:,:,0].flatten(), out[:,:,1].flatten())
    # plt.imshow(out[:,:,0])
    plt.imshow(ifft2(fftshift(out)).real)           # Some times need fftshift, sometimes doesn't
    # plt.imshow(np.fft.ifft2(out).real)
    # plt.imshow((np.abs(s_fftim)), vmin=0, vmax=3500)
    # plt.imshow((np.abs(out)), vmin=0, vmax=3500)
    # plt.imshow((np.abs(H_0._outflow[:,:,0]) + np.roll(np.abs(H_0._outflow[:,:,1]), 1)), vmin=0, vmax=2500)
    plt.show()

    '''Test frequency support calculation'''
    # H_0 = Downsample()
    # H_0.set_core_matrix(np.array([[2, 0], [-1, 1]]))
    # G_0 = Upsample()
    # G_0.set_core_matrix(np.array([[2, 0], [-1, 1]]))
    # G_0._inflow = np.random.random([512,512, 2])
    # G_0._calculate_freq_support()
    # M = len(G_0._support)
    # ncol=3
    # fig, axs = plt.subplots(int(np.ceil(M / float(ncol))), int(ncol))
    # for i in xrange(M):
    #     if M <= ncol:
    #         try:
    #             axs[i].imshow(G_0._support[i])
    #         except:
    #             pass
    #     else:
    #         axs[i//ncol, i%ncol].imshow(G_0._support[i])
    # # # axs[-1, -1].imshow(np.sum(np.stack([G_0._support[i].astype('int') for i in xrange(M)]), 0))
    # plt.imshow(np.sum(np.stack([G_0._support[i].astype('int') for i in xrange(M)]), 0))
    # plt.show()
    #



