from FilterBanks import DirectionalFilterBankDown, DirectionalFilterBankUp
from imageio import imread
import matplotlib.pyplot as plt
from numpy.fft import fftshift, fft2, ifft2, ifftshift


''' Testing'''
if __name__ == '__main__':

    '''Prepare input image'''
    im = imread('./Materials/lena_gray.png')[:,:,0]

    '''Instruction to manual phase shift'''
    # # These phase shift operations can produce fan filter effect, but introduce the need to do fftshift
    # x, y = np.meshgrid(np.arange(im.shape[0]), np.arange(im.shape[1]))
    # px = np.zeros(im.shape, dtype=np.complex)
    # px.imag = -np.pi*x
    # px = np.exp(px)
    # im = im*px

    # shift the input so that origin is at center of image
    s_fftim = fftshift(fft2(ifftshift(im)))

    '''Create filter tree'''
    P_0 = DirectionalFilterBankDown()
    U_0 = DirectionalFilterBankUp(P_0)

    outnode = P_0
    out = outnode.run(s_fftim)


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


