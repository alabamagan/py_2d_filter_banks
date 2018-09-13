from FilterBanks import Interpolation, Decimation, FilterBankNodeBase, FilterNode
from Filters1D import LPIIR8_Poly
from TwoBandFilters import TwoBandDecimation, TwoBandInterpolation
from abc import ABCMeta, abstractmethod
import numpy as np


class PresetFilterBase(FilterNode):
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

    def set_shrink(self, shrink):
        self._shrink = shrink

    def run(self, inflow):
        if not self._input_node is None:
            return self._core_function(self._input_node.run(inflow))
        else:
            return self._core_function(inflow)


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

#=====================================================================

class FanFilter(PresetFilterBase):
    _filter = None
    _E0 = None          # Analysis polyphase components
    _E1 = None          # Analysis polyphase components

    def __init__(self, inNode=None):
        super(FanFilter, self).__init__(inNode)

    def _prepare_filter(self, shape):
        if FanFilter._filter is None:
            x = np.arange(-shape//2, shape //2) * 2 * np.pi / shape
            x = np.exp(x*1j)
            z1, z2 = np.meshgrid(x, x)

            # For fan beam, one axis is inverted
            z1 = -z1

            # Polyphase components
            E0_zz, E1_zz = LPIIR8_Poly(z1*z2)
            E0_zzi, E1_zzi = LPIIR8_Poly(z1*z2**-1)

            FanFilter._E0 = E0_zz * E0_zzi
            FanFilter._E1 = E1_zz * E1_zzi
            FanFilter._filter = [FanFilter._E0 + z1 ** -1 * FanFilter._E1,
                                 FanFilter._E0 - z1 ** -1 * FanFilter._E1]
        elif FanFilter._filter[0].shape[0] != shape:
            FanFilter._filter = None
            self._prepare_filter(shape)

        self._filter = [np.copy(f) for f in FanFilter._filter]


class DiamondFilter(PresetFilterBase):
    _filter = None
    _E0 = None          # Analysis polyphase components
    _E1 = None          # Analysis polyphase components

    def __init__(self, inNode=None):
        super(DiamondFilter, self).__init__(inNode)

    def _prepare_filter(self, shape):
        if DiamondFilter._filter is None:
            x = np.arange(-shape//2, shape //2) * 2 * np.pi / shape
            x = np.exp(x*1j)
            z1, z2 = np.meshgrid(x, x)

            # Polyphase components
            E0_zz, E1_zz = LPIIR8_Poly(z1*z2)
            E0_zzi, E1_zzi = LPIIR8_Poly(z1*z2**-1)

            DiamondFilter._E0 = E0_zz * E0_zzi
            DiamondFilter._E1 = E1_zz * E1_zzi
            DiamondFilter._filter = [DiamondFilter._E0 + z1 ** -1 * DiamondFilter._E1,
                                     DiamondFilter._E0 - z1 ** -1 * DiamondFilter._E1]
        elif DiamondFilter._filter[0].shape[0] != shape:
            DiamondFilter._filter = None
            self._prepare_filter(shape)

        self._filter = [np.copy(f) for f in DiamondFilter._filter]


class CheckBoardFilter(FanFilter):
    def __init__(self, inNode=None):
        Q = np.array([[1, 1],[-1,1]])
        super(CheckBoardFilter, self).__init__(inNode)
        self.set_post_modulation_matrix(Q.T)

class ParallelogramFilter(FanFilter):
    def __init__(self, direction, inNode=None):
        r"""Mapper is as follow when applied to concatenate fan filters
            '1c'->0
            '2c'->1
            '2r'->2
            '1r'->3
        """

        R1 = np.mat([[1, 1], [0, 1]])
        R2 = np.mat([[1, -1], [0, 1]])
        R3 = np.mat([[1, 0], [1, 1]])
        R4 = np.mat([[1, 0], [-1, 1]])

        if direction == '1c':
            R = R3
        elif direction == '1r':
            R = R2
        elif direction == '2c':
            R = R1
        elif direction == '2r':
            R = R4
        else:
            raise AttributeError("Direction argument must be one of ['1c', '1r', '2c', '2r']")

        if direction == '2c' or direction == '2r':
            Q = np.mat([[1, 1],[-1,1]]).T
        else:
            Q = np.mat([[1, 1],[-1,1]])
        super(ParallelogramFilter, self).__init__(inNode)
        self.set_post_modulation_matrix(np.array(R.T))

        if direction == '1c':
            self._D = np.array(R*Q*R2)
        elif direction == '1r':
            self._D = np.array(R*Q*R3)
        elif direction == '2c':
            self._D = np.array(R*Q*R4)
        elif direction == '2r':
            self._D = np.array(R*Q*R1)



class DirectionalFilter(PresetFilterBase):
    def __init__(self, inNode=None):
        super(DirectionalFilter, self).__init__(inNode)

        self._r1 = ParallelogramFilter('2c')
        self._r2 = ParallelogramFilter('1r')
        self._r3 = ParallelogramFilter('1c')
        self._r4 = ParallelogramFilter('2r')

        self._ds = [self._r1, self._r2, self._r3, self._r4] # List for convinience

    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.shape[0] == inflow.shape[1]
        assert inflow.ndim == 3

        if inflow.shape[-1] > 4:

            pass
        else:
            t0 = self._r1.run(inflow[:,:,0])
            t1 = self._r2.run(inflow[:,:,1])
            t2 = self._r3.run(inflow[:,:,2])
            t3 = self._r4.run(inflow[:,:,3])

            self._outflow = np.concatenate([t0, t1, t2, t3], axis=-1)

            return self._outflow

    def _prepare_filter(self, shape):
        DirectionalFilter._filter = [1]
        self._filter = [1]


class DirectionalFilterBankDown(Decimation):
    def __init__(self, level=3, inNode=None):
        super(DirectionalFilterBankDown, self).__init__(inNode)

        self._f1 = FanFilter()
        self._f2 = CheckBoardFilter(self._f1)
        self._d1 = Decimation(self._f2)
        self._d1.set_core_matrix(np.mat([[1, 1],[-1,1]]) *
                                 np.mat([[1, 1],[-1,1]]))

        self._f3 = DirectionalFilter()

        self._decimation = [Decimation() for i in xrange(4)]
        for i in xrange(len(self._decimation)):
            self._decimation[i].set_core_matrix(self._f3._ds[i]._D)

    def set_shrink(self, b):
        if b:
            self._f3.hook_input(None)
        else:
            self._f3.hook_input(self._f2)

        super(DirectionalFilterBankDown, self).set_shrink(b)

    def _core_function(self, inflow):
        if self._shrink:
            temp = self._d1.run(inflow)
            temp = self._f3.run(temp[::2,::2])

            out = []
            for i, d in enumerate(self._decimation):
                out.append(d.run(temp[:,:,2*i:2*i+2]))

            out[0] = out[0][:,::2]
            out[1] = out[1][:,::2]
            out[2] = out[2][::2,:].transpose(1, 0, 2)
            out[3] = out[3][::2,:].transpose(1, 0, 2)

            return np.concatenate(out, -1)
        else:
            raise NotImplementedError()
