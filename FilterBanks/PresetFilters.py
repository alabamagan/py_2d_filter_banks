from FilterBanks import Upsample, Downsample, FilterBankNodeBase
from TwoBandFilters import TwoBandDownsample, TwoBandUpsample
from abc import ABCMeta, abstractmethod
import numpy as np


class FanDecimator(TwoBandDownsample):
    def __init__(self, inNode=None):
        super(FanDecimator, self).__init__(inNode)
        self.set_shift([0.5, 0])
        pass


class FanInterpolator(TwoBandUpsample):
    def __init__(self, inNode=None):
        super(FanInterpolator, self).__init__(inNode)
        self.set_shift([0.5, 0])
        pass


class PresetFilterBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, inNode=None):
        if not inNode is None:
            self.hook_input(inNode)
        pass


    def hook_input(self, node):
        self._in_node = node

    @abstractmethod
    def run(self, inflow):
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


class DirectionalDecimator(object):
    def __init__(self):
        super(DirectionalDecimator, self).__init__()
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


class DirectionalInterpolator(object):
    def __init__(self):
        super(DirectionalInterpolator, self).__init__()
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


class DirectionalFilterBankDown(PresetFilterBase):
    def __init__(self):
        super(DirectionalFilterBankDown, self).__init__(None)

        self._d1 = FanDecimator()
        self._d2 = FanDecimator(self._d1)
        self._d3 = DirectionalDecimator()
        self._d3.hook_input(self._d2)

    def run(self, inflow):
        return self._d3.run(inflow)


class DirectionalFilterBankUp(PresetFilterBase):
    def __init__(self):
        super(DirectionalFilterBankUp, self).__init__(None)

        self._u1 = DirectionalInterpolator()
        self._u2 = FanInterpolator(self._u1)
        self._u3 = FanInterpolator(self._u2)

    def run(self, inflow):
        return self._u3.run(inflow)