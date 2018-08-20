from FilterBanks import Upsample, Downsample, FilterBankNodeBase
from TwoBandFilters import TwoBandDownsample, TwoBandUpsample
import numpy as np

class FanDecimator(TwoBandDownsample):
    def __init__(self, inNode=None):
        super(FanDecimator, self).__init__(inNode)
        self.set_shift([0.5, 0])
        pass

class ParalleloidDecimator(Downsample):
    def __init__(self, direction, inNode=None):
        r"""
        Mapper is as follow when applied to concatenate fan filters
            '1c'->0
            '2c'->1
            '2r'->2
            '1r'->3
        """
        super(ParalleloidDecimator, self).__init__(inNode)

        if direction == '1c':
            self.set_core_matrix(np.array([[1, -1],
                                           [0, 2]]))
            pass
        elif direction == '1r':
            self.set_core_matrix(np.array([[2, 0],
                                           [1, 1]]))
            pass
        elif direction == '2c':
            self.set_core_matrix(np.array([[1, 1],
                                           [0, 2]]))
            pass
        elif direction == '2r':
            self.set_core_matrix(np.array([[2, 0],
                                           [-1, 1]]))
            pass

    def _core_function(self, inflow):
        temp = np.fft.fftshift(inflow)
        self._outflow = super(ParalleloidDecimator, self)._core_function(temp)
        self._outflow = np.fft.fftshift(self._outflow)
        return self._outflow

class ParalleloidUpsampler(Upsample):
    def __init__(self, direction, inNode=None):
        super(ParalleloidUpsampler, self).__init__(inNode)

        if direction == '1c':
            self.set_core_matrix(np.array([[1, -1],
                                           [0, 2]]))
            pass
        elif direction == '1r':
            self.set_core_matrix(np.array([[2, 0],
                                           [1, 1]]))
            pass
        elif direction == '2c':
            self.set_core_matrix(np.array([[1, 1],
                                           [0, 2]]))
            pass
        elif direction == '2r':
            self.set_core_matrix(np.array([[2, 0],
                                           [-1, 1]]))
            pass

    def _core_function(self, inflow):
        temp = np.fft.fftshift(inflow)
        self._outflow = super(ParalleloidUpsampler, self)._core_function(temp)
        self._outflow = np.fft.fftshift(self._outflow)
        return self._outflow

class DirectionalDicimator(object):
    def __init__(self):
        super(DirectionalDicimator, self).__init__()
        self._in_node = None

        self._d1c = ParalleloidDecimator('1c')    # map to 0
        self._d1r = ParalleloidDecimator('1r')    # map to 3
        self._d2c = ParalleloidDecimator('2c')    # map to 1
        self._d2r = ParalleloidDecimator('2r')    # map to 2


    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.shape[0] == inflow.shape[1]
        assert inflow.ndim == 3

        if inflow.shape[-1] > 4:

            pass
        else:
            t0 = self._d1c.run(inflow[:,:,0])
            t1 = self._d2c.run(inflow[:,:,1])
            t2 = self._d2r.run(inflow[:,:,2])
            t3 = self._d1r.run(inflow[:,:,3])

            return np.concatenate([t0, t1, t2, t3], axis=-1)

    def run(self, inflow):
        if self._in_node is None:
            return self._core_function(inflow)
        else:
            return self._core_function(self._in_node.run(inflow))

    def hook_input(self, node):
        assert isinstance(node, TwoBandDownsample)
        self._in_node = node


class DirectionalInterpolation(object):
    def __init__(self):
        super(DirectionalInterpolation, self).__init__()
        self._in_node = None

        self._i1c = ParalleloidUpsampler('1c')
        self._i1r = ParalleloidUpsampler('1r')
        self._i2c = ParalleloidUpsampler('2c')
        self._i2r = ParalleloidUpsampler('2r')


    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.shape[0] == inflow.shape[1]
        assert inflow.ndim == 3

        if inflow.shape[-1] > 8:
            pass
        else:
            t0 = self._i1c.run(inflow[:,:,:2])
            t1 = self._i2c.run(inflow[:,:,2:4])
            t2 = self._i2r.run(inflow[:,:,4:6])
            t3 = self._i1r.run(inflow[:,:,6:8])

            return np.stack([t0, t1, t2, t3], axis=-1)

    def run(self, inflow):
        if self._in_node is None:
            return self._core_function(inflow)
        else:
            return self._core_function(self._in_node.run(inflow))

    def hook_input(self, node):
        self._in_node = node



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
    p_0 = DirectionalInterpolation()
    H_0 = FanDecimator()
    H_1 = FanDecimator()
    H_1.hook_input(H_0)
    P_0.hook_input(H_1)
    p_0.hook_input(P_0)
    # G_0 = TwoBandUpsample()
    # G_0.set_shift([0.5, 0])
    # G_0.hook_input(H_1)

    outnode = P_0
    out = outnode.run(s_fftim)
    print out.shape
    print "Finished"
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
            # axs[i].imshow(ifft2(H_0._outflow[:,:,i]).real)
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
