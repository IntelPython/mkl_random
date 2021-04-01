import numpy as np

__doc__ = """
https://math.stackexchange.com/questions/351913/probability-that-a-stick-randomly-broken-in-five-places-can-form-a-tetrahedron

Choose 5 locations on a stick to break it into 6 pieces. What is the probability that these 6 pieces can be edge-lengths of a 
tetrahedron (3D symplex).

"""

__all__ = ['mc_three_piece_stick_triangle_prob', 'mc_six_piece_stick_tetrahedron_prob']

def triangle_inequality_(x1, x2, x3):
    """Efficiently finds `np.less(x1,x2+x3)*np.less(x2,x1+x3)*np.less(x3,x1+x2)`"""
    tmp_sum = x2 + x3
    res = np.less(x1, tmp_sum)   # x1 < x2 + x3
    np.add(x1, x3, out=tmp_sum)
    buf = np.less(x2, tmp_sum)   # x2 < x1 + x3
    np.logical_and(res, buf, out=res)
    np.add(x1, x2, out=tmp_sum)
    np.less(x3, tmp_sum, out=buf) # x3 < x1 + x2
    np.logical_and(res, buf, out=res)
    return res


def triangle_inequality(x1, x2, x3, out=None):
    """Efficiently finds `np.less(x1,x2+x3)*np.less(x2,x1+x3)*np.less(x3,x1+x2)`, 
       logically ending this on top of out array, if any"""
    if out is None:
        return triangle_inequality_(x1, x2, x3)
    res = out
    tmp_sum = x2 + x3
    buf = np.less(x1, tmp_sum)   # x1 < x2 + x3
    np.logical_and(res, buf, out=res)
    np.add(x1, x3, out=tmp_sum)
    np.less(x2, tmp_sum, out=buf)   # x2 < x1 + x3
    np.logical_and(res, buf, out=res)
    np.add(x1, x2, out=tmp_sum)
    np.less(x3, tmp_sum, out=buf) # x3 < x1 + x2
    np.logical_and(res, buf, out=res)
    return res


def facial_tetrahedron(x, y, z, xb, yb, zb):
    """
    Computes boolean mask for facial tetrahedron condition for six side-lengths
    This condition is necessary, but not sufficient for 3 sticks to form a tetrahedon yet,
    it needs to be supplemented with positivity of Cayley-Manger determinant.
    """
    success_mask = triangle_inequality(x, y, zb)        # x, y, zb
    triangle_inequality(x, y, zb, out = success_mask)   # x, yb, z
    triangle_inequality(xb, y, z, out = success_mask)   # xb, y, z
    triangle_inequality(xb, yb, zb, out = success_mask) # xb, yb, zb
    return success_mask


def cayley_menger_mat(x2, y2, z2, xb2, yb2, zb2):
    """
    Menger's determinant.

    If positive, there exist 4 points in R^3, with pair-wise distances squared equal to given 6 arguments.

    K. Wirth, A.S. Dreiding, Edge lengths determining tetrahedrons, Elemente der Mathematic, vol. 64 (2009) pp. 160-170.
    """
    one = np.ones_like(x2)
    zero = np.zeros_like(x2)
    mat = np.array([[zero, x2, y2, z2, one], 
                    [x2, zero, zb2, yb2, one], 
                    [y2, zb2, zero, xb2, one], 
                    [z2, yb2, xb2, zero, one], 
                    [one, one, one, one, zero]
    ]).T
    return mat


def cayley_menger_det_no_linalg(x2, y2, z2, xb2, yb2, zb2):
    """
    D(S) = 2 * x2 * xb2 * (y2 + yb2 + z2 + zb2 - x2 - xb2) +
           2 * y2 * yb2 * (z2 + zb2 + x2 + xb2 - y2 - yb2) + 
           2 * z2 * zb2 * (x2 + xb2 + y2 + yb2 - z2 - zb2) +
           (x2 - xb2) * (y2 - yb2) * (z2 - zb2) -
           (x2 + xb2) * (x2 + xb2) * (z2 + zb2)
    """
    xs = x2 + xb2
    ys = y2 + yb2
    zs = z2 + zb2
    buf1 = ys + zs
    buf1 -= xs
    buf2 = x2 * xb2
    buf1 *= buf2 # buf1 has first term, halved
    np.multiply(y2, yb2, out=buf2)
    buf3 = xs + zs
    buf3 -= ys
    buf2 *= buf3 # buf2 has second term
    buf1 += buf2 # buf1 is sum of two terms, halved
    np.multiply(z2, zb2, out=buf3)
    np.add(xs, ys, out=buf2) # reuse buf2
    buf2 -= zs
    buf3 *= buf2 # buf3 has third term
    buf1 += buf3 # buf1 is sum of 3 first terms, halved
    buf1 *= 2
    np.subtract(x2, xb2, out=buf2)
    np.subtract(y2, yb2, out=buf3)
    buf2 *= buf3
    np.subtract(z2, zb2, out=buf3)
    buf2 *= buf3
    buf1 += buf2 # buf1 is sum of 4 first terms
    np.multiply(xs, ys, out=buf3)
    buf3 *= zs
    buf1 -= buf3
    return buf1
    

def cayley_menger_cond(x2, y2, z2, xb2, yb2, zb2):
    # return np.linalg.det(cayley_menger_mat(x2, y2, z2, xb2, yb2, zb2)) > 0
    return cayley_menger_det_no_linalg(x2, y2, z2, xb2, yb2, zb2) > 0


def mc_six_piece_stick_tetrahedron_prob(rs, n):
    """
    Monte-Carlo estimate of the probability that a unit stick, randomly broken in 5 places (making 6 pieces), 
    can form a tetrahedron.

    Using provided random state instance `rs` routine generates `n` samples, and outputs the number of 
    tetrahedral 6-tuples.
    """
    u = rs.rand(6,n)
    u[0, :] = 1
    np.log(u[1], out=u[1])
    u[1] /= 5
    np.exp(u[1], out=u[1]) # np.power(u[1], 1/5, out=u[1])
    np.sqrt(u[2], out=u[2])
    np.sqrt(u[2], out=u[2])
    np.cbrt(u[3], out=u[3])
    np.sqrt(u[4], out=u[4])
    np.cumprod(u, axis=0, out=u)
    u[0] -= u[1]
    u[1] -= u[2]
    u[2] -= u[3]
    u[3] -= u[4]
    u[4] -= u[5]

    success_mask = facial_tetrahedron(u[0], u[1], u[2], u[3], u[4], u[5])
    np.square(u, out=u) # only squares enter Cayler-Manger determinant
    cm_mask = cayley_menger_cond(u[0], u[1], u[2], u[3], u[4], u[5])
    np.logical_and(success_mask, cm_mask, out=success_mask)
    
    return success_mask.sum()


def mc_three_piece_stick_triangle_prob(rs, n):
    """
    Monte-Carlo estimate of probability that a unit stick, randomly broken in 2 places (making 3 pieces),
    corresponds to a triple of sides of a triangle.

    Using provided random state instance `rs` routine generates `n` samples, and outputs the number of 
    triangular 3-tuples."""
    ws = np.sort(rs.rand(2,n), axis=0)
    x2 = np.empty(n, dtype=np.double)
    x3 = np.empty(n, dtype=np.double)

    x1 = ws[0]
    np.subtract(ws[1], ws[0], out=x2)
    np.subtract(1, ws[1], out=x3)

    return triangle_inequality_(x1, x2, x3).sum()
