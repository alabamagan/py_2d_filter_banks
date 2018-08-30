from FilterBanks import DirectionalFilterBankDown, DirectionalFilterBankUp
from imageio import imread
from numpy.fft import fftshift, fft2, ifftshift

from FilterBanks.Functions.Utility import display_subbands

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

    outnode = U_0
    out = P_0.run(im)
    # recovered = outnode.run(s_fftim)

    '''Display recovered image'''
    # plt.imshow(ifft2(fftshift(recovered)).real - ifftshift(im))
    # plt.show()

    '''Display subband components'''
    display_subbands(out, ncol=4)

