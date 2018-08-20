from FilterBanks import DirectionalDicimator, DirectionalInterpolator, FanDecimator

''' Testing'''
if __name__ == '__main__':
    from imageio import imread
    import matplotlib.pyplot as plt
    from numpy.fft import fftshift, fft2, ifft2, ifftshift

    im = imread('../../Materials/s0.tif')
    # # These phase shift operations can produce fan filter effect, but introduce the need to do fftshift
    # x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    # px = np.zeros(im.shape, dtype=np.complex)
    # px.imag = -np.pi*x
    # px = np.exp(px)
    # im = im*px

    # shift the input so that origin is at center of image
    s_fftim = fftshift(fft2(ifftshift(im)))

    '''Create filter tree'''
    P_0 = DirectionalDicimator()
    p_0 = DirectionalInterpolator()
    H_0 = FanDecimator()
    H_1 = FanDecimator()
    H_1.hook_input(H_0)
    P_0.hook_input(H_1)
    p_0.hook_input(P_0)


    outnode = P_0
    out = outnode.run(s_fftim)
    # out = outnode.run(s_fftim)


    '''Display image'''
    # plt.imshow((np.abs(out)), vmin=0, vmax=3500)
    # plt.imshow(ifft2(fftshift(out)).real - ifftshift(im))
    # plt.show()

    '''Display subband components'''
    M = out.shape[-1]
    ncol=4
    fig, axs = plt.subplots(M / ncol, ncol)
    for i in xrange(M):
        if M <= ncol:
            try:
                axs[i].imshow(ifft2(fftshift(out[:,:,i])).real, cmap='Greys_r')
                # axs[i].imshow(np.abs(fftshift(out[:,:,i])), vmin=0, vmax=2500)
                axs[i].set_title('%s'%i)
            except:
                pass
        else:
            axs[i//ncol, i%ncol].imshow(ifft2(fftshift(out[:,:,i])).real, cmap='Greys_r')
            # axs[i//ncol, i%ncol].imshow(np.abs(fftshift(out[:,:,i])), vmin=0, vmax=2500)
            axs[i//ncol, i%ncol].set_title('%s'%i)
    plt.show()

    '''Test frequency support calculation'''
    # M = len(outnode._support)
    # ncol=3
    # fig, axs = plt.subplots(int(np.ceil(M / float(ncol))), int(ncol))
    # for i in xrange(M):
    #     if M <= ncol:
    #         try:
    #             axs[i].imshow(outnode._support[i])
    #         except:
    #             pass
    #     else:
    #         axs[i//ncol, i%ncol].imshow(outnode._support[i])
    # # # axs[-1, -1].imshow(np.sum(np.stack([G_0._support[i].astype('int') for i in xrange(M)]), 0))
    # plt.imshow(np.sum(np.stack([outnode._support[i].astype('int') for i in xrange(M)]), 0))
    # plt.show()
    #
