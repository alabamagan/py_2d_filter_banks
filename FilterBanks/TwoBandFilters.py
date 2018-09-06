from FilterBanks import Interpolation, Decimation
import numpy as np

class TwoBandDecimation(Decimation):
    def __init__(self, inNode=None):
        super(TwoBandDecimation, self).__init__(inNode)

    def _core_function(self, inflow):
        r"""

        :param inflow:
        :return:
        """

        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.shape[0] == inflow.shape[1]

        if inflow.ndim == 2:
            self._outflow = super(TwoBandDecimation, self)._core_function(inflow)
            return self._outflow
        else:
            self._outflow = np.concatenate([super(TwoBandDecimation, self)._core_function(inflow[:, :, i])
                                            for i in xrange(inflow.shape[-1])], axis=2)
            return self._outflow


class TwoBandInterpolation(Interpolation):
    def __init__(self, inNode=None):
        super(TwoBandInterpolation, self).__init__(inNode)
    
    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.shape[0] == inflow.shape[1]
        assert inflow.ndim == 3
        if inflow.shape[-1] == 2:
            return super(TwoBandInterpolation, self)._core_function(inflow)
        elif inflow.shape[-1] % 2 == 0:
            l = inflow.shape[-1]
            t0 = np.copy(self._core_function(inflow[:,:,:l//2]))
            t1 = np.copy(self._core_function(inflow[:,:,l//2:l]))

            if t0.ndim == t1.ndim == 3:
                return self._core_function(np.concatenate([t0, t1], axis=-1))
            else:
                return self._core_function(np.stack([t0, t1], axis=-1))


