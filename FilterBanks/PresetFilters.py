from FilterBanks import Interpolation, Decimation, FilterBankNodeBase, FilterNode
from Filters1D import LPIIR8_Poly
from TwoBandFilters import TwoBandDecimation, TwoBandInterpolation
from abc import ABCMeta, abstractmethod
import numpy as np


class PresetFilterBase(FilterNode):
    __metaclass__ = ABCMeta
    def __init__(self, inNode=None, synthesis_mode = False):
        super(PresetFilterBase, self).__init__(inNode, synthesis_mode)
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



class FanFilter(PresetFilterBase):
    _analysis_filter = None
    _E0 = None          # Analysis polyphase components
    _E1 = None          # Analysis polyphase components

    _synthesis_filter = None
    _G0 = None
    _G1 = None

    def __init__(self, inNode=None, synthesis_mode = False):
        super(FanFilter, self).__init__(inNode, synthesis_mode)

    def _prepare_filter(self, shape):
        if FanFilter._analysis_filter is None:
            # The method introduced by Ansari and Lau, 1987, were used to construct 2D filter from 1D LPIIR filter
            x = np.arange(-shape//2, shape //2) * 2 * np.pi / shape
            x = np.exp(x*1j)
            z1, z2 = np.meshgrid(x, x)

            # For fan beam, one axis is inverted
            z1 = -z1

            # Polyphase components of 1D filters
            E0_zz, E1_zz = LPIIR8_Poly(z1*z2)
            E0_zzi, E1_zzi = LPIIR8_Poly(z1*z2**-1)

            FanFilter._E0 = E0_zz * E0_zzi
            FanFilter._E1 = E1_zz * E1_zzi
            FanFilter._analysis_filter = [(FanFilter._E0 + z1 ** -1 * FanFilter._E1)*2, # H0 = E0 + z^-1 * E1
                                          (FanFilter._E0 - z1 ** -1 * FanFilter._E1)*2] # H1 = E0 - z^-1 * E1

            H0, H1 = FanFilter._analysis_filter
            # Choose synthesis filter accordingly
            FanFilter._synthesis_filter = [2*H0 / (H0**2 - H1**2),
                                           -2*H1 / (H0**2 - H1**2)]

        elif FanFilter._analysis_filter[0].shape[0] != shape:
            FanFilter._analysis_filter = None
            self._prepare_filter(shape)

        if not self._synthesis_mode:
            self._filter = [np.copy(f) for f in FanFilter._analysis_filter]
        else:
            self._filter = [np.copy(f) for f in FanFilter._synthesis_filter]


class DiamondFilter(PresetFilterBase):
    _analysis_filter = None
    _E0 = None          # Analysis polyphase components
    _E1 = None          # Analysis polyphase components

    _synthesis_filter = None
    _G0 = None
    _G1 = None


    def __init__(self, inNode=None, synthesis_mode = False):
        super(DiamondFilter, self).__init__(inNode, synthesis_mode)

    def _prepare_filter(self, shape):
        if DiamondFilter._analysis_filter is None:
            x = np.arange(-shape//2, shape //2) * 2 * np.pi / shape
            x = np.exp(x*1j)
            z1, z2 = np.meshgrid(x, x)

            # Polyphase components
            E0_zz, E1_zz = LPIIR8_Poly(z1*z2)
            E0_zzi, E1_zzi = LPIIR8_Poly(z1*z2**-1)

            DiamondFilter._E0 = E0_zz * E0_zzi
            DiamondFilter._E1 = E1_zz * E1_zzi
            DiamondFilter._analysis_filter = [(DiamondFilter._E0 + z1 ** -1 * DiamondFilter._E1)*2,
                                              (DiamondFilter._E0 - z1 ** -1 * DiamondFilter._E1)*2]

            H0, H1 = DiamondFilter._analysis_filter
            # Choose synthesis filter accordingly
            DiamondFilter._synthesis_filter = [2*H0 / (H0**2 - H1**2),
                                           -2*H1 / (H0**2 - H1**2)]

        elif DiamondFilter._analysis_filter[0].shape[0] != shape:
            DiamondFilter._analysis_filter = None
            self._prepare_filter(shape)

        if not self._synthesis_mode:
            self._filter = [np.copy(f) for f in DiamondFilter._analysis_filter]
        else:
            self._filter = [np.copy(f) for f in DiamondFilter._synthesis_filter]


class CheckBoardFilter(FanFilter):
    def __init__(self, inNode=None, synthesis_mode = False):
        Q = np.array([[1, 1],[-1,1]])
        super(CheckBoardFilter, self).__init__(inNode, synthesis_mode)
        self.set_post_modulation_matrix(Q.T)

class ParallelogramFilter(FanFilter):
    _Q = np.mat([[1, 1],[-1,1]])
    _R1 = np.mat([[1, 1], [0, 1]])
    _R2 = np.mat([[1, -1], [0, 1]])
    _R3 = np.mat([[1, 0], [1, 1]])
    _R4 = np.mat([[1, 0], [-1, 1]])
    _samplingMatrix = [_R1*_Q.T*_R4,
                       _R2*_Q*_R3,
                       _R3*_Q*_R2,
                       _R4*_Q.T*_R1]

    def __init__(self, direction, inNode=None, synthesis_mode = False):
        r"""Mapper is as follow when applied to concatenate fan filters
            '1c'->0
            '2c'->1
            '2r'->2
            '1r'->3
        """

        Q = ParallelogramFilter._Q
        if direction == '1c':
            R = ParallelogramFilter._R3
        elif direction == '1r':
            R = ParallelogramFilter._R2
        elif direction == '2c':
            R = ParallelogramFilter._R1
        elif direction == '2r':
            R = ParallelogramFilter._R4
        else:
            raise AttributeError("Direction argument must be one of ['1c', '1r', '2c', '2r']")

        if direction == '2c' or direction == '2r':
            Q = Q.T

        super(ParallelogramFilter, self).__init__(inNode, synthesis_mode)
        self.set_post_modulation_matrix(np.array(R.T))

        if direction == '1c':
            self._D = np.array(R*Q*ParallelogramFilter._R2)
        elif direction == '1r':
            self._D = np.array(R*Q*ParallelogramFilter._R3)
        elif direction == '2c':
            self._D = np.array(R*Q*ParallelogramFilter._R4)
        elif direction == '2r':
            self._D = np.array(R*Q*ParallelogramFilter._R1)



class DirectionalFilter(PresetFilterBase):
    def __init__(self, inNode=None, synthesis_mode = False):
        super(DirectionalFilter, self).__init__(inNode)

        self._r1 = ParallelogramFilter('2c', synthesis_mode=synthesis_mode)
        self._r2 = ParallelogramFilter('1r', synthesis_mode=synthesis_mode)
        self._r3 = ParallelogramFilter('1c', synthesis_mode=synthesis_mode)
        self._r4 = ParallelogramFilter('2r', synthesis_mode=synthesis_mode)

        self._ds = [self._r1, self._r2, self._r3, self._r4] # List for convinience

    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.shape[0] == inflow.shape[1]
        assert inflow.ndim == 3

        if inflow.shape[-1] > 4:

            pass
        else:
            t0 = self._r2.run(inflow[:,:,0])
            t1 = self._r1.run(inflow[:,:,1])
            t2 = self._r4.run(inflow[:,:,2])
            t3 = self._r3.run(inflow[:,:,3])

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

        self._f3 = DirectionalFilter(self._d1)

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
            temp = self._f3.run(inflow)

            temp = [d.run(temp[:,:,2*i:2*i+2]) for i, d in enumerate(self._decimation)]
            return np.concatenate(temp, -1)


class DirectionalFilterBankUp(Interpolation):
    def __init__(self, level=3, inNode=None):
        super(DirectionalFilterBankUp, self).__init__(inNode)

        self._u1 = [Interpolation() for i in xrange(2**level)]

        self._f1 = DirectionalFilter(synthesis_mode=True)