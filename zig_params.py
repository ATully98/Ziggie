import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.signal import detrend
from matplotlib import path
from IZZI_MD_calc import calc_IZZI_MD

#########################
# Yu (2012; JGR) method #
#########################
# get normalised points
def get_Z(Xpts, Ypts, b, seg_min, seg_max):
    Xseg = Xpts[seg_min:seg_max+1]
    Yseg = Ypts[seg_min:seg_max+1]
    Zx_points = Xseg/Ypts[0]
    Zy_points = Yseg/Ypts[0]
    n = len(Xseg)

    # NRM max
    Z_NRM = Zy_points[0]
    # TRM min
    Z_TRM = Zx_points[0]
    # find b at each point. NRM loss / pTRM gained
    bi = (Z_NRM - Zy_points)/(Zx_points-Z_TRM)
    # difference from reference value
    b_p = (bi - np.abs(b))
    # weighting factor
    r = Zx_points/Zx_points[-1]
    # ensure nan
    bi[0] = np.nan

    number = n-1
    Z= np.nansum(b_p*r)/np.sqrt(number)

    return Z
#########################
# Yu (2012; JGR) method #
#########################
# get normalised points
def get_Z_star(Xpts, Ypts, Y_int, b, seg_min, seg_max):
    Xseg = Xpts[seg_min:seg_max+1]
    Yseg = Ypts[seg_min:seg_max+1]
    Zx_points = Xseg/Ypts[0]
    Zy_points = Yseg/Ypts[0]
    z_y_int = (Y_int/Ypts[0]).item()
    Z_NRM = Zy_points[0]
    Z_TRM = Zx_points[0]
    n = len(Xseg)

    # find b at each point. NRM loss / pTRM
    bi = (Z_NRM - Zy_points)/(Zx_points-Z_TRM)
    # ensure nan
    bi[0] = np.nan
    number = n-1
    bi_r = np.abs((bi-np.abs(b))*Zx_points)
    Z_star = 100*np.nansum(bi_r)/(number*z_y_int)
    return Z_star

# Find the scaled length of the best fit line
# normalise points
# def get_ziggie(Xpts, Ypts, seg_min, seg_max):
#     Xn = Xpts[seg_min:seg_max+1]
#     Xn = Xn/Xn[-1]
#     Yn = Ypts[seg_min:seg_max+1]
#     Yn = Yn/Yn[0]
#     n = len(Xn)
#     # find best fit line
#     U = detrend(Xn, type = "constant", axis = 0) # (Xi-Xbar)
#     V = detrend(Yn, type = "constant", axis = 0) # (Yi-Ybar)
#     b = np.sign(np.sum(U*V))*np.std(Yn, ddof = 1)/np.std(Xn, ddof = 1);
#     Y_int = np.mean(Yn) - b*np.mean(Xn)
#     X_int = -Y_int/b

#     # Project the data onto the best-fit line
#     Rev_x = (Yn - Y_int) / b # The points reflected about the bes-fit line
#     Rev_y = b * Xn + Y_int
#     x_prime = (Xn + Rev_x)/2   # Average the both sets to get the projected points
#     y_prime = (Yn + Rev_y)/2

#     # Get the TRM, NRM, and line lengths
#     Delta_x_prime = np.abs( np.amax(x_prime)-np.amin(x_prime) )
#     Delta_y_prime = np.abs( np.amax(y_prime)-np.amin(y_prime) )
#     Line_Len = np.sqrt(Delta_x_prime**2 + Delta_y_prime**2)

#     # Set cumulative length to 0
#     cum_len = 0.0

#     # iterate through pairs of points in Arai plot
#     for i in range(0, n-1):

#         # find the distance between the two points
#         dist = np.sqrt((Xn[i+1,0] - Xn[i,0])**2 + (Yn[i+1,0] - Yn[i,0])**2)
#         # Add to the cumulative distance
#         cum_len = cum_len + dist

#     # calculate the log of the cumulative length over the length of the best fit line
#     ziggie = np.log(cum_len/Line_Len)
#     return ziggie, cum_len, Line_Len, Xn, Yn, x_prime, y_prime

def get_gradients(Xpts, Ypts, seg):
    X_seg = Xpts[seg]
    Y_seg = Ypts[seg]
    xbar = np.mean(X_seg)
    ybar = np.mean(Y_seg)
    U = detrend(X_seg, type = "constant", axis = 0) # (Xi-Xbar)
    V = detrend(Y_seg, type = "constant", axis = 0) # (Yi-Ybar)
    b = np.sign(np.sum(U*V))*np.std(Y_seg, ddof = 1)/np.std(X_seg, ddof = 1)
    n = len(X_seg)

    sigma_b = np.sqrt( (2*np.sum(V**2)-2*(b)*np.sum(U*V)) / ( (n-2)*np.sum(U**2)) )
    beta = np.abs(sigma_b/b)


    Y_int = np.mean(Y_seg) - b*np.mean(X_seg)
    X_int = -Y_int/b

    return b, Y_int, beta

def get_beta(Xpts, Ypts, seg):
    X_seg = Xpts[seg]
    Y_seg = Ypts[seg]
    n = len(X_seg)
    xbar = np.mean(X_seg)
    ybar = np.mean(Y_seg)
    U = detrend(X_seg, type = "constant", axis = 0) # (Xi-Xbar)
    V = detrend(Y_seg, type = "constant", axis = 0) # (Yi-Ybar)
    b = np.sign(np.sum(U*V))*np.std(Y_seg, ddof = 1)/np.std(X_seg, ddof = 1)
    sigma_b = np.sqrt( (2*np.sum(V**2)-2*(b)*np.sum(U*V)) / ( (n-2)*np.sum(U**2)) )
    beta = np.abs(sigma_b/b)


    Y_int = np.mean(Y_seg) - b*np.mean(X_seg)
    X_int = -Y_int/b

    return beta


def get_points_2d(j, grad, n):
    grad = -grad
    angle = np.arctan(-grad)
    Ypts = np.arange(0,n+1,1)
    Ypts = Ypts.reshape((len(Ypts),1))/n
    Xpts = (Ypts - 1)/grad

    Ypts[1:-1:2] += j*np.sin(angle)
    Ypts[2:-1:2] -= j*np.sin(angle)
    Xpts[1:-1:2] += j*np.cos(angle)
    Xpts[2:-1:2] -= j*np.cos(angle)

    Xpts = Xpts[::-1]
    Ypts = Ypts[::-1]


    return Xpts, Ypts



def create_array(shape):
    """
    creaty array containing nan values
    Input:
            shape - desired shape of output array
    Output:
            arr - array of input shape containing nan values at every index
    """
    # create empty array and populate with nans
    arr = np.empty(shape)
    arr[:] = np.nan

    return arr


def SCAT(b, Xpts, Ypts, seg_min, seg_max, n):

    seg = np.arange(seg_min, seg_max+1,1)
    X_seg = Xpts[seg]
    Y_seg = Ypts[seg]
    xbar = np.mean(X_seg)
    ybar = np.mean(Y_seg)

    ## SCAT - uses the user input
    beta_T = 0.1
    sigma_T = beta_T*np.abs(b)
    b1 = b - 2*sigma_T
    b2 = b + 2*sigma_T

    # determine the intercepts
    a1 = ybar-b1*xbar # upper
    a2 = ybar-b2*xbar # lower
    a3 = -a2/b2 # the upper
    a4 = -a1/b1 # and lower x-axis intercepts

    C1 = np.array([[0, a2]]) # lower left corner
    C2 = np.array([[0, a1]]) # upper left corner
    C3 = np.array([[a3, 0]]) # upper right corner
    C4 = np.array([[a4, 0]]) # lower right corner
    SCAT_BOX = np.concatenate((C1, C2, C3, C4, C1),axis = 0)

    Check_points =  create_array((1,1))# the x-, y-coords of all the checks within the SCAT range
    Check_points =  np.delete(Check_points, (0), axis=0)




    # Create an array with the points to test
    if Check_points.size == 0:
        SCAT_points = (np.concatenate((X_seg, Y_seg),axis = 1)) # Add the TRM-NRM Arai plot points

    else:
        quit()

    eps = (np.finfo(float).eps)
    # add tolerance so points on line are included
    for point in SCAT_points:
        if point[0] == 0.0:
            point[0] = point[0]+eps
        if point[1] == 0.0:
            point[1] = point[1] + eps
    p = path.Path(SCAT_BOX)
    IN = p.contains_points(SCAT_points)

    SCAT = np.floor(np.sum(IN)/len(IN)) # The ratio ranges from 0 to 1, the floor command rounds down to nearest integer (i.e., rounds to 0 or 1)

    ## Get multiple SCATs - uses a hard coded range of beta thresholds

    # beta_thresh=(0.002:0.002:0.25);
    beta_thresh = np.linspace(0.01, 0.25, 100)
    nthresh = len(beta_thresh)

    tmp_SCAT = np.empty((nthresh,1))
    tmp_SCAT[:] = np.nan

    for i in range(nthresh):
        tmp_beta_T = beta_thresh[i]

        sigma_T = tmp_beta_T*np.abs(b)
        b1 = b - 2*sigma_T
        b2 = b + 2*sigma_T

        # determine the intercepts
        a1 =  ybar-b1* xbar # upper
        a2 =  ybar-b2* xbar # lower
        a3 = -a2/b2 # the upper
        a4 = -a1/b1 # and lower x-axis intercepts

        C1 = np.array([[0, a2]]) # lower left corner
        C2 = np.array([[0, a1]]) # upper left corner
        C3 = np.array([[a3, 0]]) # upper right corner
        C4 = np.array([[a4, 0]]) # lower right corner
        SCAT_BOX_mul = np.concatenate((C1, C2, C3, C4, C1),axis = 0)

        Check_points =  create_array((1,1))# the x-, y-coords of all the checks within the SCAT range
        Check_points =  np.delete(Check_points, (0), axis=0)  # the x-, y-coords of all the checks within the SCAT range

        # Create an array with the points to test
        if Check_points.size == 0:
            SCAT_points = (np.concatenate((X_seg, Y_seg),axis = 1)) # Add the TRM-NRM Arai plot points
        else:
            SCAT_points = vstack((np.concatenate((X_seg, Y_seg),axis = 1), Check_points)) # Add the TRM-NRM Arai plot points

        for point in SCAT_points:
            if point[0] == 0.0:
                point[0] = point[0]+eps
            if point[1] == 0.0:
                point[1] = point[1] + eps

        p = path.Path(SCAT_BOX_mul)
        IN = p.contains_points(SCAT_points)
        tmp_SCAT[i] = np.floor(np.sum(IN)/len(IN))  # The ratio ranges from 0 to 1, the floor command rounds down to nearest integer (i.e., rounds to 0 or 1)

    multi_SCAT = tmp_SCAT
    multi_SCAT_beta = beta_thresh
    return SCAT


def get_ecdf(ecdf_data):
    numbers = ecdf_data.to_list()
    n_p = int(len(numbers))
    y_p = np.linspace(1/n_p, 1, num=n_p)

    numbers.insert(0,0.0)
    numbers.sort()
    y_p = np.insert(y_p,0,0.0)

    return numbers, y_p


def get_IZZI_MD(Xpts,Ypts,Treatment,seg_min,seg_max):
    if seg_min == 0:
        seg_min = 1
    IZZI_MD = calc_IZZI_MD(Xpts,Ypts,Treatment,seg_min,seg_max)
    return np.float64(IZZI_MD)




def AraiCurvature(x,y):
    """
    Function for calculating the radius of the best fit circle to a set of
    x-y coordinates.
    Paterson, G. A., (2011), A simple test for the presence of multidomain
    behaviour during paleointensity experiments, J. Geophys. Res., doi: 10.1029/2011JB008369

    Inputs:
            x - array of shape (n, 1) containg Arai plot x points
            y - array of shape (n, 1) containg Arai plot y points
    Output:
            parameters - array of shape (4, 1):
                        parameters[0,0] = k
                        parameters[1,0] = a
                        parameters[2,0] = b
                        parameters[3,0] = SSE (goodness of fit)
    """
    # Reshape vectors for suitable input
    x = np.reshape(x, (len(x), 1))
    y = np.reshape(y, (len(y), 1))

    # Normalize vectors
    x = x /np.amax(x)
    y = y/np.amax(y)

    # Provide the initial estimate
    # This will be used if number of points <=3
    E1 = TaubinSVD(np.concatenate((x,y),axis = 1))
    estimates = np.array((E1[0,2], E1[0,0], E1[0,1]))

    if len(x) > 3:
        # Determine the iterative solution
        #This needs at least 4 points for calculating the variance
        E2 = LMA(np.concatenate((x,y),axis=1), E1)
        estimates = np.array((E2[2], E2[0], E2[1]))

    else:
        E2 = E1.reshape(E1.size,)


    # Define the function to be minimized and calculate the SSE
    func = lambda v: np.sum((np.sqrt((x-v[1])**2+(y-v[2])**2)-v[0])**2)
    SSE = func(estimates)
    func_rms = lambda v: np.sqrt(np.mean(((np.sqrt((x-v[1])**2+(y-v[2])**2)-v[0])**2)))
    RMS = func_rms(estimates)

    if E2[0] <= np.mean(x) and E2[1] <= np.mean(y):
        k = -1/E2[2];
    else:
        k = 1/E2[2]


    parameters = np.array(([k], [E2[0]], [E2[1]], [SSE], [RMS]))

    return parameters # Arai plot curvature


def TaubinSVD(XY):
    """

    Algebraic circle fit by Taubin
    G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
                Space Curves Defined By Implicit Equations, With
                Applications To Edge And Range Image Segmentation",
            IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)

    Input:  XY - array of shape (n, 2) of coordinates of n points

    Output: Par - (1, 3) array containing a, b, R for the fitting circle: center (a,b) and radius R
                            Par[0,0] - a
                            Par[0,1] - b
                            Par[0,2] - R


    Note: this is a version optimized for stability, not for speed
    """
    centroid = np.mean(XY, axis = 0)   # the centroid of the data set

    X = XY[:,0:1] - centroid[0]  #  centering data
    Y = XY[:,1:2] - centroid[1]  #  centering data
    Z = X*X + Y*Y
    Zmean = np.mean(Z)
    Z0 = (Z-Zmean)/(2*(Zmean**(0.5)))
    ZXY = np.concatenate((Z0, X, Y),axis=1)
    U, S, V = np.linalg.svd(ZXY,compute_uv = True)

    A = V[2]
    A[0] = A[0]/(2*(Zmean**(0.5)))

    A = (np.concatenate((A, np.array(([-Zmean*A[0]])) ),axis = 0)).reshape((4,1))
    Par = np.concatenate((( -np.transpose(A[1:3])/A[0]/2+centroid, (np.sqrt(A[1]*A[1]+A[2]*A[2]-4*A[0]*A[3])/np.abs(A[0])/2).reshape(1,1)   )),axis = 1)

    return Par   #  TaubinSVD



def LMA(XY,ParIni):
    """
    Geometric circle fit (minimizing orthogonal distances)
    based on the Levenberg-Marquardt scheme in the
    "algebraic parameters" A,B,C,D  with constraint B*B+C*C-4*A*D=1
    N. Chernov and C. Lesort, "Least squares fitting of circles",
    J. Math. Imag. Vision, Vol. 23, 239-251 (2005)

    Input:  XY - array of shape (n, 2) of coordinates of n points
            ParIni = array containing (a, b, R) is the initial guess (supplied by user)

    Output: Par - (3, ) array containing a, b, R for the fitting circle: center (a,b) and radius R
                            Par[0] - a
                            Par[1] - b
                            Par[2] - R
    """

    factorUp = 10
    factorDown = 0.04
    lamb0 = 0.01
    epsilon = 0.000001
    IterMAX = 50
    AdjustMax = 20
    Xshift = 0
    Yshift = 0
    dX = 1
    dY = 0

    n = np.shape(XY)[0]	  # number of data points

    # starting with the given initial guess

    anew = ParIni[0,0] + Xshift
    bnew = ParIni[0,1] + Yshift

    Anew = 1/(2*ParIni[0,2])
    aabb = anew*anew + bnew*bnew
    Fnew = (aabb - ParIni[0,2]*ParIni[0,2])*Anew
    Tnew = np.arccos(-anew/np.sqrt(aabb))
    if bnew > 0:
        Tnew = 2*np.pi - Tnew

    VarNew = VarCircle(XY,ParIni)

    #	 initializing lambda and iter
    lamb = lamb0
    finish = 0

    for iter in range(IterMAX):

        Aold = Anew
        Fold = Fnew
        Told = Tnew
        VarOld = VarNew

        H = np.sqrt(1+4*Aold*Fold)
        aold = -H*np.cos(Told)/(Aold+Aold) - Xshift
        bold = -H*np.sin(Told)/(Aold+Aold) - Yshift
        Rold = 1/np.abs(Aold+Aold)

        #  computing moments
        DD = 1 + 4*Aold*Fold
        D = np.sqrt(DD)
        CT = np.cos(Told)
        ST = np.sin(Told)

        H11=0
        H12=0
        H13=0
        H22=0
        H23=0
        H33=0
        F1=0
        F2=0
        F3=0

        for i in range(n):
            Xi = XY[i,0] + Xshift
            Yi = XY[i,1] + Yshift
            Zi = Xi*Xi + Yi*Yi
            Ui = Xi*CT + Yi*ST
            Vi =-Xi*ST + Yi*CT

            ADF = Aold*Zi + D*Ui + Fold
            SQ = np.sqrt(4*Aold*ADF + 1)
            DEN = SQ + 1
            Gi = 2*ADF/DEN
            FACT = 2/DEN*(1 - Aold*Gi/SQ)
            DGDAi = FACT*(Zi + 2*Fold*Ui/D) - Gi*Gi/SQ
            DGDFi = FACT*(2*Aold*Ui/D + 1)
            DGDTi = FACT*D*Vi

            H11 = H11 + DGDAi*DGDAi
            H12 = H12 + DGDAi*DGDFi
            H13 = H13 + DGDAi*DGDTi
            H22 = H22 + DGDFi*DGDFi
            H23 = H23 + DGDFi*DGDTi
            H33 = H33 + DGDTi*DGDTi

            F1 = F1 + Gi*DGDAi
            F2 = F2 + Gi*DGDFi
            F3 = F3 + Gi*DGDTi

        for adjust in range(AdjustMax):

            # Cholesly decomposition

            G11 = np.sqrt(H11 + lamb)
            G12 = H12/G11
            G13 = H13/G11
            G22 = np.sqrt(H22 + lamb - G12*G12)
            G23 = (H23 - G12*G13)/G22
            G33 = np.sqrt(H33 + lamb - G13*G13 - G23*G23)

            D1 = F1/G11
            D2 = (F2 - G12*D1)/G22
            D3 = (F3 - G13*D1 - G23*D2)/G33

            dT = D3/G33
            dF = (D2 - G23*dT)/G22
            dA = (D1 - G12*dF - G13*dT)/G11

            # updating the parameters
            Anew = Aold - dA
            Fnew = Fold - dF
            Tnew = Told - dT

            if 1+4*Anew*Fnew < epsilon and lamb>1:
                Xshift = Xshift + dX
                Yshift = Yshift + dY

                H = np.sqrt(1+4*Aold*Fold)
                aTemp = -H*np.cos(Told)/(Aold+Aold) + dX
                bTemp = -H*np.sin(Told)/(Aold+Aold) + dY
                rTemp = 1/np.abs(Aold+Aold)

                Anew = 1/(rTemp + rTemp)
                aabb = aTemp*aTemp + bTemp*bTemp
                Fnew = (aabb - rTemp*rTemp)*Anew
                Tnew = np.arccos(-aTemp/np.sqrt(aabb))
                if bTemp > 0:
                    Tnew = 2*np.pi - Tnew

                VarNew = VarOld
                break

            if 1+4*Anew*Fnew < epsilon:
                lamb = lamb * factorUp
                continue

            DD = 1 + 4*Anew*Fnew
            D = np.sqrt(DD)
            CT = np.cos(Tnew)
            ST = np.sin(Tnew)

            GG = 0;

            for i in range(n):
                Xi = XY[i,0] + Xshift
                Yi = XY[i,1] + Yshift
                Zi = Xi*Xi + Yi*Yi
                Ui = Xi*CT + Yi*ST

                ADF = Anew*Zi + D*Ui + Fnew
                SQ = np.sqrt(4*Anew*ADF + 1)
                DEN = SQ + 1
                Gi = 2*ADF/DEN
                GG = GG + Gi*Gi

            VarNew = GG/(n-3)

            H = np.sqrt(1+4*Anew*Fnew)
            anew = -H*np.cos(Tnew)/(Anew+Anew) - Xshift
            bnew = -H*np.sin(Tnew)/(Anew+Anew) - Yshift
            Rnew = 1/np.abs(Anew+Anew)

            # checking if improvement is gained
            if VarNew <= VarOld:	  #   yes, improvement
                progress = (np.abs(anew-aold) + np.abs(bnew-bold) + np.abs(Rnew-Rold))/(Rnew+Rold)
                if progress < epsilon:
                    Aold = Anew
                    Fold = Fnew
                    Told = Tnew
                    VarOld = VarNew
                    finish = 1
                    break

                lamb = lamb * factorDown
                break
            else:					 #   no improvement
                lamb = lamb * factorUp
                continue

        if finish == 1:
            break

    H = np.sqrt(1+4*Aold*Fold)
    Par_1 = -H*np.cos(Told)/(Aold+Aold) - Xshift
    Par_2 = -H*np.sin(Told)/(Aold+Aold) - Yshift
    Par_3 = 1/np.abs(Aold+Aold)
    Par = np.array(([Par_1,Par_2,Par_3]))

    return Par  # LMA


def VarCircle(XY,Par):

    """
    Fuction computing the sample variance of distances from data points (XY) to the circle Par = [a b R]
    Inputs:
            XY - array of shape (n, 2) containing X and Y points
            Par - array of shape (1, 3) containing (a, b, R) is the initial guess, where circle center (a,b) and radius R

    Output:
            Var - float value for the variance
    """

    n = np.shape(XY)[0]	  # number of data points

    Dx = XY[:,0:1] - Par[0,0]
    Dy = XY[:,1:2] - Par[0,1]
    D = np.sqrt(Dx*Dx + Dy*Dy) - Par[0,2]

    Var = (np.transpose(D)@D/(n-3))[0,0]

    return Var  #  VarCircle

def intersection(a, b, radius, p2x, p2y):

    """ find the two points where a secant intersects a circle """
    dx, dy = p2x - a, p2y - b
    j = dx**2 + dy**2
    k = 2 * (dx * (p2x- a) + dy * (p2y - b))
    l = (p2x - a)**2 + (p2y -b)**2 - radius**2

    discriminant = k**2 - 4 * j * l
#     assert (discriminant > 0), 'Not a secant!'

    t1 = (-k + discriminant**0.5) / (2 * j)
    t2 = (-k - discriminant**0.5) / (2 * j)

    return (dx * t1 + p2x, dy * t1 + p2y), (dx * t2 + p2x, dy * t2 + p2y)

def closer(a,b,p1,p2):
    dist1 = np.sqrt( (p1[0] - a)**2 + (p1[1] - b)**2 )
    dist2 = np.sqrt( (p2[0] - a)**2 + (p2[1] - b)**2 )
    if dist1 > dist2:
        return p2
    else:
        return p1

def AraiArc(k, a, b, Xn1, Yn1, Xn2,Yn2):
    r = np.abs(1/k)

    p1, p2 = intersection(a, b, r, Xn2, Yn2)
    p = closer(Xn2,Yn2,p1,p2)
    x2=p[0]
    y2=p[1]

    p1, p2 = intersection(a, b, r, Xn1, Yn1)
    p = closer(Xn1,Yn1,p1,p2)
    x1=p[0]
    y1=p[1]

    # vectors 1 and 2
    vec1 = [a-x1, b-y1]/np.linalg.norm([a-x1, b-y1])
    vec2 = [a-x2, b-y2]/np.linalg.norm([a-x2, b-y2])

    # angle between vectors
    angle = np.arctan2(vec1[0] * vec2[1] - vec1[1] * vec2[0], vec1[0] * vec2[0] + vec1[1] * vec2[1])

    if (angle < 0):
        angle += 2*np.pi
    if k<0:
        angle = 2*np.pi - angle
    # arc for angle in radians

    arc = r*angle
    return arc, (x1, y1), (x2,y2)

# Find the scaled length of the best fit line
# normalise points
def get_zigzag(Xn, Yn, seg_min, seg_max):

    n = len(Xn)
    # find best fit line
    U = detrend(Xn, type = "constant", axis = 0) # (Xi-Xbar)
    V = detrend(Yn, type = "constant", axis = 0) # (Yi-Ybar)
    b = np.sign(np.sum(U*V))*np.std(Yn, ddof = 1)/np.std(Xn, ddof = 1);
    Y_int = np.mean(Yn) - b*np.mean(Xn)
    X_int = -Y_int/b

    # Project the data onto the best-fit line
    Rev_x = (Yn - Y_int) / b # The points reflected about the bes-fit line
    Rev_y = b * Xn + Y_int
    x_prime = (Xn + Rev_x)/2   # Average the both sets to get the projected points
    y_prime = (Yn + Rev_y)/2

    # Get the TRM, NRM, and line lengths
    Delta_x_prime = np.abs( np.amax(x_prime)-np.amin(x_prime) )
    Delta_y_prime = np.abs( np.amax(y_prime)-np.amin(y_prime) )
    Line_Len = np.sqrt(Delta_x_prime**2 + Delta_y_prime**2)

    # Set cumulative length to 0
    cum_len = 0.0

    # iterate through pairs of points in Arai plot
    for i in range(0, n-1):

        # find the distance between the two points
        dist = np.sqrt((Xn[i+1,0] - Xn[i,0])**2 + (Yn[i+1,0] - Yn[i,0])**2)
        # Add to the cumulative distance
        cum_len = cum_len + dist

    # calculate the log of the cumulative length over the length of the best fit line
    ziggie = np.log(cum_len/Line_Len)
    return ziggie, cum_len, Line_Len

# Find the scaled length of the best fit line
# normalise points
def get_ziggie(Xpts, Ypts, seg_min, seg_max):

    Xn = Xpts[seg_min:seg_max+1]
    Xn = Xn/np.amax(Xn)
    Yn = Ypts[seg_min:seg_max+1]
    Yn = Yn/ np.amax(Yn)
    ymax_idx = np.where(Yn == np.max(Yn))
    xmax_idx = np.where(Xn == np.max(Xn))
    k_prime, a, b, SSE, RMS = AraiCurvature(Xn,Yn)
#     print(f"k_prime: {k_prime}")
    if (np.abs(k_prime) <= 1e-3) or (np.isnan(k_prime)):
        ziggie, cum_len, Line_Len = get_zigzag(Xn, Yn, seg_min, seg_max)
        return ziggie, cum_len, Line_Len
    else:
        arc, point1, point2 = AraiArc(k_prime.item(), a.item(), b.item(), Xn[0][0],Yn[0][0],Xn[-1][0], Yn[-1][0],  )
        n = len(Xn)

#         # Set cumulative length to 0
        cum_len = 0.0

        # iterate through pairs of points in Arai plot
        for i in range(0, n-1):

            # find the distance between the two points
            dist = np.sqrt((Xn[i+1,0] - Xn[i,0])**2 + (Yn[i+1,0] - Yn[i,0])**2)
            # Add to the cumulative distance
            cum_len = cum_len + dist

        # calculate the log of the cumulative length over the length of the best fit line
        ziggie = np.log(cum_len/arc)
        return ziggie, cum_len, arc
