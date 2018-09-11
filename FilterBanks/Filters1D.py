import numpy as np

def LPIIR8(z):
    r"""LIPIIR8(z) --> np.ndarray
    Description
    ===========
        Return the digital frequency fitler extracted from [1] which was an extension based on [2]
        The filter is given by the equation

        .. math::
            H_0(z) = K \frac{(1+z^{-1})I(\omega_1)I(\omega_2)I(\omega_3)I(\omega_4)}
            {(1+\alpha_1^2 z^{-2})(1+\alpha_1^{-2} z^{-2})(1+\alpha_2^2 z^{-2}) (1 + \alpha_2^{-2}z^{-2}}`

        where :math:`I(\omega_i) = 1 - 2\cos{\omega_i} z^{-1} + z^{-2}`

    Reference
    =========
    [1] Park, Sang-Il, Mark JT Smith, and Russell M. Mersereau. "Improved structures of maximally decimated directional
        filter banks for spatial image analysis." IEEE Transactions on Image Processing 13.11 (2004): 1424-1431.

    [2] Smith, Mark JT, and Steven L. Eddins. "Analysis/synthesis techniques for subband image coding."
        IEEE Transactions on Acoustics, Speech, and Signal Processing 38.8 (1990): 1446-1456.


    Parameters
    ==========

    :param z: Z of z transform
    :return: np.ndarrary
    """

    def I(omega, z):
        return 1 - 2*np.cos(omega)*z**(-1.) + z**(-2.)

    omega_1 = 2.57994607
    omega_2 = 2.21432487
    omega_3 = 1.91248705
    omega_4 = 1.72625369
    a_1 = 2.24245241
    a_2 = 1.14369193
    K = 0.20146905
    return K * ((1 + z**(-1)) * I(omega_1, z) * I(omega_2, z) * I(omega_3, z) * I(omega_4, z)) / \
           ((1 + a_1 **2 * z**-2) * (1 + a_1 **-2 * z**-2) * (1 + a_2**2 * z**-2) * (1 + a_2**-2 * z**-2))


def LPIIR8_Poly(z):
    r"""LPIIR8_Poly --> np.ndarray, np.ndarray

    Description
    ===========
        The polyphase representation of LIPIIR8(z) (see also `LPIIR8(z)`). The coefficents of the
        terms are listed as follow:

        ============    =============
        |Power of z|    |Coefficient|
        ============    =============
        0               1
        -1              4.87627657
        -2              12.94597729
        -3              23.28196200
        -4              30.77687426
        -5              30.77687426
        -6              23.28196200
        -7              12.94597729
        -8              4.87627567
        -9              1
        ============    =============

        The denomenator is given by:

        .. math::
            ((1 + a_1 **2 * z**-2) * (1 + a_1 **-2 * z**-2) * (1 + a_2**2 * z**-2) * (1 + a_2**-2 * z**-2))


    Parameters
    ==========
    :param: z: Z of z transform
    :return: np.ndarray, np.ndarray
    """
    a_1 = 2.24245241
    a_2 = 1.14369193
    K = 0.20146905

    coef = np.array([1,
                     4.87627657,
                     12.94597729,
                     23.28196200,
                     30.77687426,
                     30.77687426,
                     23.28196200,
                     12.94597729,
                     4.87627567,
                     1])

    E_0 = np.sum(np.stack([c * z **(-i*1) for i, c in enumerate(coef[::2])], -1), -1)
    E_1 = np.sum(np.stack([c * z **(-i*1) for i, c in enumerate(coef[1::2])], -1), -1)

    denom = ((1 + a_1 **2 * z**-1) * (1 + a_1 **-2 * z**-1) * (1 + a_2**2 * z**-1) * (1 + a_2**-2 * z**-1))

    E_0, E_1 = [e * K / denom for e in [E_0, E_1]]
    return E_0, E_1