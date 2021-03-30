# Author: Javad Amirian
# Email: amiryan.j@gmail.com
# from https://github.com/lessandro/nbis/blob/rel-4.0.0/misc/nistmeshlab/meshlab/src/external/tsai-30b3/cal_tran.c
# original paper: https://www.dca.ufrn.br/~lmarcos/courses/visao/artigos/CameraCalibrationTsai.pdf
import math

SQR = lambda x: x ** 2
CUB = lambda x: x ** 3
SQRT = lambda x: math.sqrt(x)
CBRT = lambda x: 0 if (x == 0) else ((x ** (1 / 3.0)) if (x > 0) else (-(-x ** (1 / 3.0))))
hypot = lambda x, y: SQRT(x * x + y * y)
SINCOS = lambda x: (math.sin(x), math.cos(x))

SQRT3 = 1.732050807568877293527446341505872366943


class CameraParameters:
    def __init__(self):
        self.Ncx = 0  # /* [sel]     Number of sensor elements in camera's x direction   */
        self.Nfx = 0  # /* [pix]     Number of pixels in frame grabber's x direction     */
        self.dx = 0  # /* [mm/sel]  X dimension of camera's sensor element (in mm)      */
        self.dy = 0  # /* [mm/sel]  Y dimension of camera's sensor element (in mm)      */
        self.dpx = 0  # /* [mm/pix]  Effective X dimension of pixel in frame grabber     */
        self.dpy = 0  # /* [mm/pix]  Effective Y dimension of pixel in frame grabber     */
        self.Cx = 0  # /* [pix]     Z axis intercept of camera coordinate system        */
        self.Cy = 0  # /* [pix]     Z axis intercept of camera coordinate system        */
        self.sx = 0  # /* []        Scale factor to compensate for any error in dpx     */


class CalibrationConstants:
    def __init__(self):
        self.f = 0  # /* [mm]          */
        self.kappa1 = 0  # /* [1/mm^2]      */
        self.p1 = 0  # /* [1/mm]        */
        self.p2 = 0  # /* [1/mm]        */

        self.Tx = 0  # /* [mm]          */
        self.Ty = 0  # /* [mm]          */
        self.Tz = 0  # /* [mm]          */

        self.Rx = 0  # /* [rad]	        */
        self.Ry = 0  # /* [rad]	        */
        self.Rz = 0  # /* [rad]	        */

        self.r1 = 0  # /* []            */
        self.r2 = 0  # /* []            */
        self.r3 = 0  # /* []            */
        self.r4 = 0  # /* []            */
        self.r5 = 0  # /* []            */
        self.r6 = 0  # /* []            */
        self.r7 = 0  # /* []            */
        self.r8 = 0  # /* []            */
        self.r9 = 0  # /* []            */

    def calc_rr(self):
        sa, ca = SINCOS(self.Rx)
        sb, cb = SINCOS(self.Ry)
        sg, cg = SINCOS(self.Rz)

        self.r1 = cb * cg
        self.r2 = cg * sa * sb - ca * sg
        self.r3 = sa * sg + ca * cg * sb
        self.r4 = cb * sg
        self.r5 = sa * sb * sg + ca * cg
        self.r6 = ca * sb * sg - cg * sa
        self.r7 = -sb
        self.r8 = cb * sa
        self.r9 = ca * cb


# ************************************************************************
# /* convert from distorted sensor to undistorted sensor plane coordinates */
# ************************************************************************
def __distorted_to_undistorted_sensor_coord__(Xd, Yd, cc: CalibrationConstants):
    distortion_factor = 1 + cc.kappa1 * (SQR(Xd) + SQR(Yd))
    Xu = Xd * distortion_factor
    Yu = Yd * distortion_factor
    return Xu, Yu


# /***********************************************************************\
# * This routine performs an inverse perspective projection to determine	*
# * the position of a point in world coordinates that corresponds to a 	*
# * given position in image coordinates.  To constrain the inverse	*
# * projection to a single point the routine requires a Z world	 	*
# * coordinate for the point in addition to the X and Y image coordinates.*
# \***********************************************************************/
def image_coord_to_world_coord(Xfd, Yfd, zw, cp: CameraParameters, cc: CalibrationConstants):
    # /* convert from image to distorted sensor coordinates */
    Xd = cp.dpx * (Xfd - cp.Cx) / cp.sx
    Yd = cp.dpy * (Yfd - cp.Cy)

    # /* convert from distorted sensor to undistorted sensor plane coordinates */
    Xu, Yu = __distorted_to_undistorted_sensor_coord__(Xd, Yd, cc)

    # /* calculate the corresponding xw and yw world coordinates	 */
    # /* (these equations were derived by simply inverting	 */
    # /* the perspective projection equations using Macsyma)	 */
    common_denominator = ((cc.r1 * cc.r8 - cc.r2 * cc.r7) * Yu +
                          (cc.r5 * cc.r7 - cc.r4 * cc.r8) * Xu -
                          cc.f * cc.r1 * cc.r5 + cc.f * cc.r2 * cc.r4)

    xw = (((cc.r2 * cc.r9 - cc.r3 * cc.r8) * Yu +
           (cc.r6 * cc.r8 - cc.r5 * cc.r9) * Xu -
           cc.f * cc.r2 * cc.r6 + cc.f * cc.r3 * cc.r5) * zw +
          (cc.r2 * cc.Tz - cc.r8 * cc.Tx) * Yu +
          (cc.r8 * cc.Ty - cc.r5 * cc.Tz) * Xu -
          cc.f * cc.r2 * cc.Ty + cc.f * cc.r5 * cc.Tx) / common_denominator

    yw = -(((cc.r1 * cc.r9 - cc.r3 * cc.r7) * Yu +
            (cc.r6 * cc.r7 - cc.r4 * cc.r9) * Xu -
            cc.f * cc.r1 * cc.r6 + cc.f * cc.r3 * cc.r4) * zw +
           (cc.r1 * cc.Tz - cc.r7 * cc.Tx) * Yu +
           (cc.r7 * cc.Ty - cc.r4 * cc.Tz) * Xu -
           cc.f * cc.r1 * cc.Ty + cc.f * cc.r4 * cc.Tx) / common_denominator

    return xw, yw


# /************************************************************************/
# /*
#        This routine converts from undistorted to distorted sensor coordinates.
#        The technique involves algebraically solving the cubic polynomial
#             Ru = Rd * (1 + kappa1 * Rd**2)
#        using the Cardan method.
#        Note: for kappa1 < 0 the distorted sensor plane extends out to a maximum
#              barrel distortion radius of  Rd = sqrt (-1/(3 * kappa1)).
# 	     To see the logic used in this routine try graphing the above polynomial
# 	     for positive and negative kappa1's
# */
def undistorted_to_distorted_sensor_coord(Xu, Yu, cc: CalibrationConstants):
    if ((Xu == 0) and (Yu == 0)) or (cc.kappa1 == 0):
        Xd = Xu
        Yd = Yu
        return Xd, Yd

    Ru = hypot(Xu, Yu)  # /* SQRT(Xu*Xu+Yu*Yu) */
    c = 1 / cc.kappa1
    d = -c * Ru

    Q = c / 3
    R = -d / 2
    D = CUB(Q) + SQR(R)

    if D >= 0:  # /* one real root */
        D = SQRT(D)
        S = CBRT(R + D)
        T = CBRT(R - D)
        Rd = S + T

        if Rd < 0:
            Rd = SQRT(-1 / (3 * cc.kappa1))
            print("\nWarning: undistorted image point to distorted image point mapping limited by\n")
            print("         maximum barrel distortion radius of %lf\n", Rd)
            print("         (Xu = %lf, Yu = %lf) -> (Xd = %lf, Yd = %lf)\n\n",
                  Xu, Yu, Xu * Rd / Ru, Yu * Rd / Ru)

    else:  # /* three real roots */
        D = SQRT(-D)
        S = CBRT(hypot(R, D))
        T = math.atan2(D, R) / 3
        sinT, cosT = SINCOS(T)

        # /* the larger positive root is    2*S*cos(T)                   */
        # /* the smaller positive root is   -S*cos(T) + SQRT(3)*S*sin(T) */
        # /* the negative root is           -S*cos(T) - SQRT(3)*S*sin(T) */

        Rd = -S * cosT + SQRT3 * S * sinT  # /* use the smaller positive root */

    lmbd = Rd / Ru
    Xd = Xu * lmbd
    Yd = Yu * lmbd
    return Xd, Yd


# /***********************************************************************\
# * This routine takes the position of a point in world coordinates [mm]	*
# * and determines the position of its image in image coordinates [pix].	*
# \***********************************************************************/
def world_coord_to_image_coord(xw, yw, zw, cp: CameraParameters, cc: CalibrationConstants):
    # /* convert from world coordinates to camera coordinates */
    xc = cc.r1 * xw + cc.r2 * yw + cc.r3 * zw + cc.Tx
    yc = cc.r4 * xw + cc.r5 * yw + cc.r6 * zw + cc.Ty
    zc = cc.r7 * xw + cc.r8 * yw + cc.r9 * zw + cc.Tz

    # /* convert from camera coordinates to undistorted sensor plane coordinates */
    Xu = cc.f * xc / zc
    Yu = cc.f * yc / zc

    # /* convert from undistorted to distorted sensor plane coordinates */
    Xd, Yd = undistorted_to_distorted_sensor_coord(Xu, Yu, cc)

    # /* convert from distorted sensor plane coordinates to image coordinates */
    Xf = Xd * cp.sx / cp.dpx + cp.Cx
    Yf = Yd / cp.dpy + cp.Cy

    return Xf, Yf
