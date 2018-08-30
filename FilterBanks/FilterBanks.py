import numpy as np
from abc import ABCMeta, abstractmethod


class FilterBankNodeBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, inNode=None):
        self._input_node = None
        self._inflow = None
        self._referencing_node = []
        self._core_matrix = None
        self._coset_vectors = []
        self._shift = 0
        self.hook_input(inNode)
        pass

    def __del__(self):
        if self._input_node != None:
            assert issubclass(type(self._input_node), FilterBankNodeBase)
            if self in self._input_node._referencing_node:
                self._input_node._referencing_node.remove(self)

    @abstractmethod
    def _core_function(self, inNode):
        return inNode

    def run(self, input):
        if self._input_node is None:
            return self._core_function(input)
        else:
            return self._core_function(self._input_node.run(input))

    def set_core_matrix(self, mat):
        assert isinstance(mat, np.ndarray)

        self._core_matrix = mat
        self._calculate_coset()

    def hook_input(self, filter):
        if filter == None:
            self._input_node = None
            return

        assert issubclass(type(filter), FilterBankNodeBase)
        self._input_node = filter

    def input(self):
        return self._input_node

    def _calculate_coset(self):
        assert not self._core_matrix is None

        self._coset_vectors = []
        M = int(np.abs(np.linalg.det(self._core_matrix)))
        inv_coremat = np.linalg.inv(self._core_matrix.T)    # coset vectors always calculated in Fourier space

        # For 2D case
        for i in xrange(-M, M):
            for j in xrange(-M, M):
                # Costruct coset vector
                v = np.array([i,j], dtype=np.float)
                f = inv_coremat.dot(v)
                if np.all((0<=f) & (f < 1.)):
                    self._coset_vectors.append(v.astype('int'))

    @staticmethod
    def _frequency_modulation(freq_im, factor):
        r"""Modulate the frequency by the input factor vector.

        """
        assert isinstance(freq_im, np.ndarray)

        s = freq_im.shape[0]

        if isinstance(factor, float):
            assert 0 < factor < 1

            shift = int(np.round(s * factor))
            return np.roll(np.roll(freq_im, shift, axis=0), shift, axis=1)
        elif isinstance(factor, np.ndarray):
            assert factor.ndim == 1
            assert factor.size == 2

            yshift, xshift = [int(np.round(factor[i] * s)) for i in xrange(2)]
            return np.roll(np.roll(freq_im, xshift, axis=0), yshift, axis=1)
        else:
            raise TypeError("Input must be a float number or np.ndarray!")

    def set_shift(self, factor):
        if not (isinstance(factor, np.ndarray)
                or isinstance(factor, float)
                or isinstance(factor, list)
                or isinstance(factor, tuple)):
            raise TypeError("Input must be a float number or np.ndarray or a tuple/list!")

        if isinstance(factor, tuple) or isinstance(factor, list):
            assert len(factor) == 2
            factor = np.array(factor)

        self._shift = factor

    @staticmethod
    def periodic_modulus_2d(arr, x_range, y_range):
        r"""
        Description
        ===========

        Modular the range of the input 2D array of vector to desired range [a, b].
        The input shape should by (X, Y, 2).

        .. math::
            f(y) = \begin{cases}
                    y & \\text{if}  & y \in [a, b] \\
                    y-a-n(b-a+1) & \\text{if} & \\text{otherwise}
                   \end{cases}


        In general, :math:`[a, b]` is a one of the partitions of :math:`\mathbb{Z}` denoted as:
            :math:`\text{Par}(Z;[a, b]) = \{[a-n(b-a+1), b-n(b-a+1)]; n\in \mathbb{Z}\}`

        When :math:`n = 0`, the element range is :math:`[a, b]`. The idea is to find the closest partition,
        defined by n, and translate the number to its unit cell, then do modulus.

        """
        assert isinstance(arr, np.ndarray)
        assert arr.ndim==3, "Array shape should be (X_shape, Y_shape, 2)"
        assert arr.shape[2] == 2, "Array shape should be (X_shape, Y_shape, 2)"
        assert len(x_range) == len(y_range) == 2
        assert x_range[1] > x_range[0] and y_range[1] > y_range[0]

        mx = np.invert((x_range[0] <= arr[:, :, 0]).astype('bool') & (arr[:, :, 0] <= x_range[1]).astype('bool'))
        my = np.invert((y_range[0] <= arr[:, :, 1]).astype('bool') & (arr[:, :, 1] <= y_range[1]).astype('bool'))
        arx = arr[:,:,0]
        ary = arr[:,:,1]
        rx = x_range[1] - x_range[0] + 1
        ry = y_range[1] - y_range[0] + 1
        Nx = np.floor((arx[mx] - x_range[0]) / rx)
        Ny = np.floor((ary[my] - y_range[0]) / ry)

        arx[mx] = arx[mx] - Nx*rx
        ary[my] = ary[my] - Ny*ry
        return arr

    @staticmethod
    def unitary_modulus_2d(arr, xrange, yrange):
        assert isinstance(arr, np.ndarray)
        assert arr.ndim==3, "Array shape should be (X_shape, Y_shape, 2)"
        assert arr.shape[2] == 2, "Array shape should be (X_shape, Y_shape, 2)"
        assert len(xrange) == len(yrange) == 2
        assert xrange[1] > xrange[0] and yrange[1] > yrange[0]

        arr[np.invert((xrange[0] <= arr[:,:,0]) & (arr[:,:,0] <= xrange[1]))] = 0
        arr[np.invert((yrange[1] <= arr[:,:,1]) & (arr[:,:,1] <= yrange[1]))] = 0
        return arr


class Downsample(FilterBankNodeBase):
    def __init__(self, inNode=None):
        FilterBankNodeBase.__init__(self, inNode)
        self._outflow = None
        self._core_matrix = np.array([[1, -1],
                                      [1,  1]])
        self._calculate_coset()

    def _core_function(self, inflow):
        r"""Decimator core function.

        The decimation follows the equation:

        .. math ::
            X_k(\omega)=\frac{1}{|\text{det}(M)|} \sum_{m\in \mathcal{N} (M^T)} X(M^{-T}(\omega - s^T m))}

        where:
            :math:`\omega` is the frequency index,
            :math:`m` is the coset vector and
            :math:`s` is the frequency space size vector.

        The algorithm relies on the rearrangement of the index :math:`\omega`, which is equivalent to the
        down-sampling/decimation in the frequency space. Note that the algorithm assumed the origin of the
        input image locates at the center of the image space. It is recommended to use the `np.roll`
        method to translate the origin.

        """
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.ndim == 2
        assert inflow.shape[0] == inflow.shape[1]

        # if not complex, assume x-space input, do fourier transform
        if not np.any(np.iscomplex(inflow)):
            # it is also necessary to shift the origin to the center of the image.
            self._inflow = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(inflow)))
        else:
            self._inflow = np.copy(inflow)

        if np.any(self._shift != 0):
            self._inflow = self._frequency_modulation(self._inflow, self._shift)

        s = inflow.shape[0]

        u, v = np.meshgrid(np.arange(s) - s//2, np.arange(s)-s//2)
        self._uv = np.stack([u, v], axis=-1)    # temp
        omega = np.stack([u, v], axis=-1)

        # Number of bands for the given core matrix to achieve critical sampling.
        M = int(np.abs(np.linalg.det(self._core_matrix)))

        # Bands are differed by coset vector calculated from the transpose of the core_matrix
        # Side Note:
        #   In the original design, it should be M^T, but because numpy has a row major
        #   design, therefore, the matrix has to be transposed before applying.
        omega = [(omega - s*v).dot(np.linalg.inv(self._core_matrix)) for v in self._coset_vectors]

        # Periodic modulus
        omega = [FilterBankNodeBase.periodic_modulus_2d(o, [-s//2, s//2-1], [-s//2, s//2-1]) for o in omega]

        # N bands of output
        outflow = [np.zeros(self._inflow.shape, dtype=self._inflow.dtype)
                   for i in xrange(M)]
        for i in xrange(inflow.shape[0]):
            for j in xrange(inflow.shape[1]):
                for k, o in enumerate(omega):
                    if o[i,j, 0] % 1 == 0 and o[i,j,1] % 1 == 0:
                        # Note that the matrix are caculate in x, y convention while in numpy it has a [y, x]
                        # convention
                        try:
                            outflow[k][i,j] = self._inflow[int(o[i,j,1] + s//2),
                                                           int(o[i,j,0] + s//2)] \
                                              / float(M)
                        except:
                            print i, j
                            pass

        self._omega = omega # temp
        self._outflow = np.stack(outflow, -1)
        if np.any(self._shift != 0):
            self._outflow = np.fft.fftshift(self._outflow)
        #     self._outflow = self._frequency_modulation(self._outflow, self._shift - 1.)
        return self._outflow

    def get_lower_subband(self):
        assert not self._outflow is None
        return self._outflow[:,:,0]

    def get_higher_subband(self):
        assert not self._outflow is None
        return self._outflow[:,:,-1]

    def get_subband(self, index):
        assert not self._outflow is None
        assert index < self._outflow.shape[-1]
        return self._outflow[:,:,int(index)]


class Upsample(FilterBankNodeBase):
    def __init__(self, inNode=None):
        FilterBankNodeBase.__init__(self, inNode)

        self._core_matrix = np.array([[1, -1],
                                      [1,  1]])
        self._calculate_coset()

    def _core_function(self, inflow):
        r"""Upsample/Interpolation core function

        The interpolation is done in frequency space following the equation:

        .. math::
            X_u(\omega)=\sum_{k} X_k(M^{-T}\omega + s^T m_k}

        :param inflow:
        :return:
        """
        assert isinstance(inflow, np.ndarray), "Input must be numpy array"
        assert inflow.ndim == 3
        assert inflow.shape[0] == inflow.shape[1]
        # assert np.all(np.iscomplex(inflow))

        self._inflow = np.copy(inflow)
        self._calculate_freq_support()
        if np.any(self._shift != 0):
            self._inflow = np.fft.fftshift(self._inflow)

        s = inflow.shape[0]

        u, v = np.meshgrid(np.arange(s) - s//2, np.arange(s)-s//2)
        self._uv = np.stack([u, v], axis=-1)    # temp
        omega = np.stack([u, v], axis=-1)

        # Bands are differed by coset vector calculated from the transpose of the core_matrix
        # Side Note:
        #   In the original design, it should be M^T, but because numpy has a row major
        #   design, therefore, the matrix has to be transposed before applying.
        omega = [(omega).dot(self._core_matrix) + s*v for v in self._coset_vectors]

        # Periodic modulus
        omega = [FilterBankNodeBase.periodic_modulus_2d(o, [-s//2, s//2-1], [-s//2, s//2-1]) for o in omega]

        # Number of bands for the given core matrix to achieve critical sampling.
        M = int(np.abs(np.linalg.det(self._core_matrix)))

        outflow = np.zeros([self._inflow.shape[0], self._inflow.shape[1]], dtype=np.complex)
        for i in xrange(inflow.shape[0]):
            for j in xrange(inflow.shape[1]):
                for m in xrange(M):
                    for k, o in enumerate(omega):
                        if o[i,j, 0] % 1 == 0 and o[i,j,1] % 1 == 0:
                            try:
                                # Note that the matrix are caculate in x, y convention while in numpy it has a [y, x]
                                # convention
                                outflow[i,j] += self._inflow[int(o[i,j,1] + s//2),
                                                             int(o[i,j,0] + s//2), m] * \
                                                self._support[m][i, j]
                            except:
                                pass
        self._omega = omega # temp
        if np.any(self._shift != 0):
            outflow = self._frequency_modulation(outflow, self._shift)
            self._support = [self._frequency_modulation(np.fft.fftshift(s), self._shift) for s in self._support]
            # outflow = np.fft.fftshift(outflow)
        self._outflow = outflow
        #     self._outflow = np.fft.fftshift(self._outflow)
        return outflow

    def _calculate_coset(self):
        """
        Description
        -----------
          Expression for up-sampling/interpolation of the input signal is given by:

          t-domain.

        :return:
        """
        assert not self._core_matrix is None

        super(Upsample, self)._calculate_coset()
        inv_core_mat = np.linalg.inv(self._core_matrix)
        self._coset_vectors = [self._core_matrix.dot(inv_core_mat.dot(v)) for v in self._coset_vectors]

    def _calculate_freq_support(self):
        assert not self._inflow is None

        s = self._inflow.shape[0]

        # Build the support with zero offset vector
        u, v = np.meshgrid(np.arange(2*s) - s, np.arange(2*s) - s)
        sup_1 = (self._core_matrix[0, 0] * u + self._core_matrix[1, 0] * v < s//2) & \
                (self._core_matrix[0, 0] * u + self._core_matrix[1, 0] * v >= -s//2)
        sup_2 = (self._core_matrix[0, 1] * u + self._core_matrix[1, 1] * v < s//2) & \
                (self._core_matrix[0, 1] * u + self._core_matrix[1, 1] * v >= -s//2)
        support = (sup_1 & sup_2).astype('int')[s//2:s//2+s, s//2:s//2+s]

        cvs = [V.dot(np.linalg.inv(self._core_matrix)) for V in self._coset_vectors]
        self._support = [np.roll(
                            np.roll(
                                support, int(np.round(V[0] * s)), axis=1),
                            int(np.round(V[1] * s)), axis=0) for V in cvs]

