from PresetFilters import *

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

        self._f1 = DirectionalFilter(synthesis_mode=True)
        self._u1 = [Interpolation() for i in xrange(2**(level - 1))]
        for i, u in enumerate(self._u1):
            u.set_core_matrix(self._f1._ds[i]._D)

        self._u2 = Interpolation()
        self._u2.set_core_matrix(np.mat([[1, 1],[-1,1]]) *
                                 np.mat([[1, 1],[-1,1]]))
        self._f2 = CheckBoardFilter(self._u2, synthesis_mode=True)
        self._f3 = FanFilter(self._f2, synthesis_mode=True)

    def _core_function(self, inflow):
        assert isinstance(inflow, np.ndarray)
        s = inflow.shape

        if self._shrink:
            if inflow.shape[-1] > 8:
                pass
            else:
                # Expand subbands
                out = np.zeros([s[0], s[1]*2, s[2]], dtype=inflow.dtype)
                out[:,::2, :4] = inflow[:,:,:4]
                out[::2,:, 4:] = inflow[:,:,4:].transpose(1, 0, 2)

                # Upsample
                temp = []
                for i, u in enumerate(self._u1):
                    temp.append(u.run(out[:,:,2*i:2*i+2]))
                out = np.concatenate(temp, -1)

                # Filter
                out = self._f1.run(out)

                # Expand again
                ss = out.shape
                temp = np.zeros([ss[0]*2, ss[1]*2, out.shape[-1]], dtype=out.dtype)
                temp[::2,::2, :] = out

                out = self._f3.run(temp)

                self._outflow = out
                return self._outflow
        else:
            # Up sample
            out = []
            for i, u in enumerate(self._u1):
                out.append(u.run(inflow[:,:,2*i:2*i+2]))
            out = np.concatenate(out, -1)

            # Filter
            out = self._f1.run(out)
            out = self._f3.run(out)

            self._outflow = out
            return self._outflow

