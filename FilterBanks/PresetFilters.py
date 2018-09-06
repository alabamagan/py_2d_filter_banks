from FilterBanks import Interpolation, Decimation, FilterBankNodeBase
from TwoBandFilters import TwoBandDecimation, TwoBandInterpolation
from abc import ABCMeta, abstractmethod
import numpy as np


class FanDecimator(TwoBandDecimation):
    def __init__(self, inNode=None):
        super(FanDecimator, self).__init__(inNode)
        self.set_shift([0.5, 0])
        pass


class FanInterpolator(TwoBandInterpolation):
    def __init__(self, inNode=None):
        super(FanInterpolator, self).__init__(inNode)
        self.set_shift([0.5, 0])
        pass


class PresetFilterBase(FilterBankNodeBase):
    __metaclass__ = ABCMeta
    def __init__(self, inNode=None):
        super(PresetFilterBase, self).__init__(inNode)
        if not inNode is None:
            self.hook_input(inNode)

        self._shrink = False
        pass


    def hook_input(self, node):
        self._input_node = node

    @staticmethod
    def _alternate_sampling(inarr, mode='col'):
        assert isinstance(inarr, np.ndarray)

        if mode == 'col':
            s = np.array(inarr.shape)
            s[1] /= 2
            out = np.zeros(s, dtype=inarr.dtype)
            out[::2,:] = np.copy(inarr[::2,::2])
            out[1::2,:] = np.copy(inarr[1::2,1::2])
            return out
        elif mode == 'row':
            s = np.array(inarr.shape)
            s[0] /= 2
            out = np.zeros(s , dtype=inarr.dtype)
            out[:,::2] = np.copy(inarr[::2,::2])
            out[:,1::2] = np.copy(inarr[1::2,1::2])
            return out

    @staticmethod
    def _alternate_zeropadding(inarr, mode='col'):
        assert isinstance(inarr, np.ndarray)


        if mode == 'col':
            s = np.array(inarr.shape)
            s[1] *= 2
            out = np.zeros(s, dtype=inarr.dtype)
            out[::2,::2] = np.copy(inarr[::2,:])
            out[1::2, 1::2] = np.copy(inarr[1::2, :])
            return out
        elif mode == 'row':
            s = np.array(inarr.shape)
            s[0] *= 2
            out = np.zeros(s, dtype=inarr.dtype)
            out[::2,::2] = np.copy(inarr[:,::2])
            out[1::2,1::2] = np.copy(inarr[:,1::2])
            return out

    @abstractmethod
    def _core_function(self, inflow):
        pass

    def set_shrink(self, shrink):
        self._shrink = shrink

    def run(self, inflow):
        if not self._input_node is None:
            return self._core_function(self._input_node.run(inflow))
        else:
            return self._core_function(inflow)

class ParalleloidDecimator(Decimation):
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


class ParalleloidUpsampler(Interpolation):
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


class DirectionalDecimator(PresetFilterBase):
    def __init__(self, inNode=None):
        super(DirectionalDecimator, self).__init__(inNode)

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

            if not self._shrink:
                self._outflow = np.concatenate([t0, t1, t2, t3], axis=-1)
            else:
                t0 = self._alternate_sampling(t0, 'col')
                t1 = self._alternate_sampling(t1, 'col')
                t2 = self._alternate_sampling(t2, 'row').transpose(1, 0, 2)
                t3 = self._alternate_sampling(t3, 'row').transpose(1, 0, 2)
                self._outflow = np.concatenate([t0, t1, t2, t3], axis=-1)

            return self._outflow

    def run(self, inflow):
        if self._input_node is None:
            return self._core_function(inflow)
        else:
            return self._core_function(self._input_node.run(inflow))

    def hook_input(self, node):
        assert isinstance(node, TwoBandDecimation) or node is None
        self._input_node = node


class DirectionalInterpolator(PresetFilterBase):
    def __init__(self):
        super(DirectionalInterpolator, self).__init__()
        self._input_node = None

        self._i1c = ParalleloidUpsampler('1c')
        self._i1r = ParalleloidUpsampler('1r')
        self._i2c = ParalleloidUpsampler('2c')
        self._i2r = ParalleloidUpsampler('2r')

    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.ndim == 3

        if not self._shrink:
            assert inflow.shape[0] == inflow.shape[1]

        if inflow.shape[-1] > 8:
            pass
        else:
            if not self._shrink:
                t0 = self._i1c.run(inflow[:,:,:2])
                t1 = self._i2c.run(inflow[:,:,2:4])
                t2 = self._i2r.run(inflow[:,:,4:6])
                t3 = self._i1r.run(inflow[:,:,6:8])
                self._outflow = np.stack([t0, t1, t2, t3], axis=-1)

            else:
                t0 = self._alternate_zeropadding(inflow[:,:,:2], 'col')
                t1 = self._alternate_zeropadding(inflow[:,:,2:4], 'col')
                t2 = self._alternate_zeropadding(inflow[:,:,4:6].transpose(1,0,2), 'row')
                t3 = self._alternate_zeropadding(inflow[:,:,6:8].transpose(1,0,2), 'row')

                print [t.shape for t in [t0, t1, t2, t3]]
                t0 = self._i1c.run(t0)
                t1 = self._i2c.run(t1)
                t2 = self._i2r.run(t2)
                t3 = self._i1r.run(t3)
                self._outflow = np.stack([t0, t1, t2, t3], axis=-1)

            return self._outflow

    def run(self, inflow):
        if self._input_node is None:
            return self._core_function(inflow)
        else:
            return self._core_function(self._input_node.run(inflow))

    def hook_input(self, node):
        self._input_node = node


class DirectionalFilterBankDown(PresetFilterBase):
    def __init__(self):
        super(DirectionalFilterBankDown, self).__init__(None)

        self._d1 = FanDecimator()
        self._d2 = FanDecimator(self._d1)
        self._d3 = DirectionalDecimator(self._d2)

    def set_shrink(self, shrink):
        super(DirectionalFilterBankDown, self).set_shrink(shrink)
        self._d3.set_shrink(shrink)
        # Rearrange pipeline
        if self._shrink:
            self._d3.hook_input(None)
        else:
            self._d3.hook_input(self._d2)


    def _core_function(self, inflow):
        if not self._shrink:
            return self._d3.run(inflow)
        else:
            return self._d3.run(self._d2.run(inflow)[::2,::2])


class DirectionalFilterBankUp(PresetFilterBase):
    def __init__(self, inNode=None):
        super(DirectionalFilterBankUp, self).__init__(inNode)

        # Note that preset fan interpolator is a two-band upsampler
        # it runs recursively until there are only only one last layer
        self._u1 = DirectionalInterpolator()
        self._u2 = FanInterpolator(self._u1)

    def set_shrink(self, shrink):
        super(DirectionalFilterBankUp, self).set_shrink(shrink)
        self._u1.set_shrink(shrink)
        if shrink:
            self._u2.hook_input(None)
        else:
            self._u2.hook_input(self._u1)

    def _core_function(self, inflow):
        if not self._shrink:
            return self._u2.run(inflow)
        else:
            assert isinstance(inflow, np.ndarray)
            t = self._u1.run(inflow)
            s = np.array(t.shape)
            s[0] *=2
            s[1] *=2
            temp = np.zeros(s, dtype=t.dtype)
            temp[::2,::2] = t
            return self._u2.run(temp)