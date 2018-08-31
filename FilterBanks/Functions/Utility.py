from numpy.fft import fftshift, ifft2
import numpy as np
import matplotlib.pyplot as plt

def display_subbands(image, ncol=2, display_freq=False, cmap='Greys_r'):
    ncol = int(ncol)
    assert ncol > 0
    assert isinstance(image, np.ndarray)

    if image.ndim == 3:
        M = image.shape[-1]
        fig, axs = plt.subplots(int(np.ceil(M / float(ncol))), ncol)
        for i in xrange(M):
            if M <= ncol:
                try:
                    if display_freq:
                        axs[i].imshow(np.abs(fftshift(image[:,:,i])), vmin=0, vmax=2500, cmap=cmap)
                    else:
                        axs[i].imshow(ifft2(fftshift(image[:,:,i])).real, cmap=cmap)
                    axs[i].set_title('%s'%i)
                except:
                    pass
            else:
                if display_freq:
                    axs[i//ncol, i%ncol].imshow(np.abs(fftshift(image[:,:,i])), vmin=0, vmax=2500, cmap=cmap)
                else:
                    axs[i//ncol, i%ncol].imshow(ifft2(fftshift(image[:,:,i])).real, cmap=cmap)
                axs[i//ncol, i%ncol].set_title('%s'%i)
        plt.show()
    elif image.ndim == 2:
        plt.imshow(image, cmap=cmap)
        plt.show()

def display_images(image, ncol=2, cmap='Greys_r', range=None):
    ncol = int(ncol)
    assert ncol > 0
    assert isinstance(image, np.ndarray)

    if image.ndim == 3:
        M = image.shape[-1]
        fig, axs = plt.subplots(int(np.ceil(M / float(ncol))), ncol)
        for i in xrange(M):
            if M <= ncol:
                try:
                    if range is None:
                        axs[i].imshow(np.real(image[:,:,i]), cmap=cmap)
                    else:
                        axs[i].imshow(np.real(image[:,:,i]), cmap=cmap, vmin=range[0], vmax=range[1])
                    axs[i].set_title('%s'%i)
                    axs[i].set_axis_off()
                except:
                    pass
            else:
                if range is None:
                    axs[i//ncol, i%ncol].imshow(np.real(image[:,:,i]), cmap=cmap)
                else:
                    axs[i//ncol, i%ncol].imshow(np.real(image[:,:,i]), cmap=cmap, vmin=range[0], vmax=range[1])
                axs[i//ncol, i%ncol].set_title('%s'%i)
                axs[i//ncol, i%ncol].set_axis_off()
        fig.subplots_adjust(left=None, right=None, top=0.9, bottom=None, wspace=None, hspace=None)
        plt.show()
    elif image.ndim == 2:
        plt.imshow(np.real(image), cmap=cmap)
        plt.show()