## FUNCIONES

from numpy import *
from numpy.linalg import *
from scipy import optimize
import sympy as sp
from concurrent.futures import ProcessPoolExecutor

# Calcular la k equivalente ----------------------------------

def k_eff(k_mat, esp_mat):
    den = 0
    k_eff_lon = 0

    for n in range(0,len(k_mat)):
        den += esp_mat[n]/k_mat[n]

        k_eff_lon += k_mat[n]*esp_mat[n]/sum(esp_mat)

    k_eff_t = sum(esp_mat)/den

    return array([k_eff_t, k_eff_lon])

# Calcular el alpha equivalente -------------------------------

def alpha_eff(k, e, rho, c):

    rho_eq = 0
    c_eq = 0

    for n in range(0,len(rho)):

        rho_eq += rho[n]*e[n]/sum(e)
        c_eq += c[n]*e[n]/sum(e)
    
    alpha = k/(rho_eq*c_eq)

    return alpha

# Caso - 1 : disipación homogénea ANALÍTICO ----------------------------

def T_case1analytic(x, k, Q1):

    T1a = zeros(len(x))
    L = 0.5
    q = Q1/(L*0.3*0.02)

    for i in range(0, len(x)):

        T1a[i] = -(q/k)*((x[i]**2)/2) + (q/k)*(L/2)*x[i] + 273.15

    return T1a

# Caso - 1 : disipación homogénea NUMÉRICO ----------------------------

def T_case1(x, k, Q):
    N = len(x)
    A = zeros([N, N])
    b = zeros(N)

    q = Q/(N-1)/((x[2]-x[1])*0.3*0.02)

    A[0,0] = 1
    A[N-1,N-1] = 1
    b[0] = 273.15
    b[N-1] = 273.15

    for i in range(1,N-1):
        A[i,i-1] = 1
        A[i,i] = -2
        A[i,i+1] = 1

        b[i] = -(q/k)*(x[2]-x[1])**2

    T1 = matmul(inv(A),b)
    
    return T1

# Caso - 2 : disipación en heaters ANALÍTICO ----------------------------

def T_case2analytic(x, k, Q, A):

    T2a = zeros(len(x))
    L = 0.5
    q = Q/(0.1*0.3*0.02)

    for i in range(0, len(x)):
        if x[i]<= 0.05:
            C1A = Q/(k*A) 
            T2a[i] = C1A*x[i] + 273.15
            Tp2 = T2a[i]

        elif x[i]>0.05 and x[i] <= 0.15:
            C1B = Q/(k*A) + q/k*0.05
            C2B = Tp2 + q/k*(0.05)**2/2 - C1B*0.05
            T2a[i] = -q/k*(x[i]**2/2) + C1B*x[i] + C2B
            Tp3 = T2a[i]

        elif x[i]>0.15 and x[i]<0.35:
            T2a[i] = Tp3

        elif x[i]>=0.35 and x[i]<0.45:
            C1C = -Q/(k*A) +q/k*0.45
            C2C = Tp3 + q/k*(0.35)**2/2- C1C*0.35
            T2a[i] = -q/k*(x[i]**2/2) + C1C*x[i] + C2C

        elif x[i]>= 0.45:
            C1D = -Q/(k*A) 
            C2D = 273.15 + Q/(k*A)*L
            T2a[i] = C1D*x[i] + C2D

    return T2a


# Caso - 2 : disipación en heaters NUMÉRICO ----------------------------

def T_case2(x, k, Q):

    N = len(x)
    A = zeros([N, N])
    b = zeros(N)

    dx = x[2]-x[1]
    q = Q/(0.1/dx)/(dx*0.3*0.02)

    A[0,0] = 1
    A[N-1,N-1] = 1
    b[0] = 273.15
    b[N-1] = 273.15

    for i in range(1,N-1):
        if (x[i]>=0.05 and x[i]<=0.15) or (x[i]>=0.35 and x[i]<=0.45):
            A[i,i-1] = 1
            A[i,i] = -2
            A[i,i+1] = 1

            b[i] = -(q/k)*(x[2]-x[1])**2

        elif x[i]<0.05 or x[i]>0.45:
            A[i,i-1] = 1
            A[i,i] = -2
            A[i,i+1] = 1

            b[i] = 0

        elif x[i]>0.15 and x[i]<0.35:
            A[i,i-1] = -1
            A[i,i] = 1

            b[i] = 0

    T2 = matmul(inv(A),b)
    
    return T2


# Caso - 3 : pérdidas por radiación NUMÉRICO LINEAL ----------------------------

def T_case3(x, k, Q, h_rad):

    N = len(x)
    A = zeros([N, N])
    b = zeros(N)

    dx = x[2]-x[1]
    q = Q/(0.1/dx)/(dx*0.3*0.02)

    A[0,0] = 1
    A[N-1,N-1] = 1
    b[0] = 273.15
    b[N-1] = 273.15

    for i in range(1,N-1):
        if (x[i]>=0.05 and x[i]<=0.15) or (x[i]>=0.35 and x[i]<=0.45): # Heaters
            A[i,i-1] = 1
            A[i,i] = -2
            A[i,i+1] = 1

            b[i] = -(q/k)*(dx)**2

        elif x[i]<0.05 or x[i]>0.45 or (x[i]>0.15 and x[i]<0.35):
            A[i,i-1] = 1
            A[i,i] = -2 - 2*h_rad/(k*0.02)*(dx)**2
            A[i,i+1] = 1

            b[i] = -2*h_rad/(k*0.02)*273.15*(dx)**2


    T3 = matmul(inv(A),b)
            
    return T3

# Caso - 3 : pérdidas por radiación NUMÉRICO NO LINEAL ----------------------------

def T_case3_nolineal(x, k, Q, sigma, eps, T3):

     N = len(x)

     dx = x[2]-x[1]
     q = Q/(0.1/dx)/(dx*0.3*0.02)

     T0 = zeros(N)
     T0[:] = T3

     def f(y):
        fun = zeros(N)
        fun[0] = y[0] - 273.15
        fun[N-1] = y[N-1] - 273.15

        for i in range(1,N-1):
            if (x[i]>=0.05 and x[i]<=0.15) or (x[i]>=0.35 and x[i]<=0.45): # Heaters

                fun[i] = y[i-1] + y[i+1] - 2*y[i] + (q/k)*(dx)**2

            elif x[i]<0.05 or x[i]>0.45 or (x[i]>0.15 and x[i]<0.35):

                fun[i] = y[i-1] + y[i+1] - 2*y[i] - 2*sigma*eps*(y[i]**4 - 273.15**4)/(k*0.02)*(dx**2)

        return fun


     T3nl = optimize.fsolve(f, T0)

     return  T3nl

 # Caso - 4 : TRANSITORIO ----------------------------------------------------

def T_case4(t, x, k, Q, sigma, eps, alpha, T3):

     Nx = len(x)
     Nt = len(t)

     dx = x[2] - x[1]
     dt = t[2] - t[1]
     q = Q/(0.1/dx)/(dx*0.3*0.02)

     T0x = zeros(Nx)
     T0x[:] = T3

     T4 = zeros([Nt, Nx])
     T4[0,:] = 273.15
     T4t = zeros(Nx)
     T4t[:] = 273.15

     def f(y):
        fun = zeros(Nx)
        fun[0] = y[0] - 273.15
        fun[Nx-1] = y[Nx-1] - 273.15

        for i in range(1,Nx-1):
            if (x[i]>=0.05 and x[i]<=0.15) or (x[i]>=0.35 and x[i]<=0.45): # Heaters

                fun[i] = (y[i-1] + y[i+1] - 2*y[i])/(dx**2) + (q/k) - (1/alpha)*(y[i]-T4t[i])/dt

            elif x[i]<0.05 or x[i]>0.45:

                fun[i] = (y[i-1] + y[i+1] - 2*y[i])/(dx**2) - 2*sigma*eps*(y[i]**4 - 273.15**4)/(k*0.02) - (1/alpha)*(y[i]-T4t[i])/dt

            elif x[i]>0.15 and x[i]<0.35:

                fun[i] = (y[i-1] + y[i+1] - 2*y[i])/(dx**2) - 2*sigma*eps*(y[i]**4 - 273.15**4)/(k*0.02) - (1/alpha)*(y[i]-T4t[i])/dt
    
        return fun

     for i in range(1, Nt):
        T4[i,:] = optimize.fsolve(f, T0x)
        T4t = T4[i,:]
        if round(T4t[int(Nx/2)],1) == round(T3[int(Nx/2)],1):
            print(t[i])
            break

     return T4

  # Caso - 5 : TRANSITORIO CON TERMOSTATOS ----------------------------------------------------

def T_case5(t, x, k, Q, sigma, eps, alpha, T3):

     Nx = len(x)
     Nt = len(t)

     dx = x[2] - x[1]
     dt = t[2] - t[1]
     q = Q/(0.1/dx)/(dx*0.3*0.02)

     T0x = zeros(Nx)
     T0x[:] = T3

     T5 = zeros([Nt, Nx])
     T5[0,:] = 273.15
     T5t = zeros(Nx)
     T5t[:] = 273.15

     T5sensor = 273.15
     Heater = 'True'

     def f(y):
        fun = zeros(Nx)
        fun[0] = y[0] - 273.15
        fun[Nx-1] = y[Nx-1] - 273.15

        for i in range(1,Nx-1):
            if (x[i]>=0.05 and x[i]<=0.15) or (x[i]>=0.35 and x[i]<=0.45): # Heaters
                
                if Heater == 'True':

                    fun[i] = (y[i-1] + y[i+1] - 2*y[i])/(dx**2) + (q/k) - (1/alpha)*(y[i]-T5t[i])/dt

                elif Heater == 'False':

                    fun[i] = (y[i-1] + y[i+1] - 2*y[i])/(dx**2) - (1/alpha)*(y[i]-T5t[i])/dt

            elif x[i]<0.05 or x[i]>0.45 or (x[i]>0.15 and x[i]<0.35):

                fun[i] = (y[i-1] + y[i+1] - 2*y[i])/(dx**2) - 2*sigma*eps*(y[i]**4 - 273.15**4)/(k*0.02) - (1/alpha)*(y[i]-T5t[i])/dt

        return fun

     for i in range(1, Nt):
        T5[i,:] = optimize.fsolve(f, T0x)
        T5t = T5[i,:]
        T5sensor = T5[i,int(Nx/2)]
        if T5sensor <= 283.15:
           Heater = 'True'  

        elif T5sensor >= 285.15:
           Heater = 'False'

     return T5

  # Caso - 6 : BIDIMENSIONAL X-Z ----------------------------------------------------

def T_case6xz(x, y, k, Q, sigma, eps, T3):

     N = len(x)
     Ny = len(y)

     dx = x[2]-x[1]
     dy = y[2]-y[1]
     q = Q/((0.1**2)/(dx*dy))/(dx*dy*0.02)

     T0 = zeros([N, Ny])

     for j in range(0,Ny):
        T0[:,j] = T3
     T0 = T0.flatten()

     def f(T):
        fun = zeros([N,Ny])
        T = T.reshape(N,Ny)
        fun[0,:] = T[0,:] - 273.15
        fun[N-1,:] = T[N-1,:] - 273.15

        for i in range(1,N-1):
                for j in range(0,Ny):

                    if (x[i]>=0.05 and x[i]<=0.15) or (x[i]>=0.35 and x[i]<=0.45): # Heaters
                        if y[j]>0.1 and y[j]<0.2:
                            fun[i,j] = (T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + (T[i,j+1] + T[i,j-1] - 2*T[i,j])/(dy**2) + (q/k)

                        elif j == 0:
                            fun[i,j] =(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + (T[i,j+1] - T[i,j])/(dy**2) - 2*sigma*eps*(T[i,j]**4 - 273.15**4)/(k*0.02)

                        elif j == (Ny-1):
                            fun[i,j] =(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + (T[i,j-1] - T[i,j])/(dy**2) - 2*sigma*eps*(T[i,j]**4 - 273.15**4)/(k*0.02)

                        else:
                             fun[i,j] =(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + (T[i,j+1] + T[i,j-1] - 2*T[i,j])/(dy**2) - 2*sigma*eps*(T[i,j]**4 - 273.15**4)/(k*0.02)

                    elif x[i]<0.05 or x[i]>0.45 or (x[i]>0.15 and x[i]<0.35):
                        if j == 0:
                            fun[i,j] =(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + (T[i,j+1] - T[i,j])/(dy**2) - 2*sigma*eps*(T[i,j]**4 - 273.15**4)/(k*0.02)

                        elif j == (Ny-1):
                            fun[i,j] =(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + (T[i,j-1] - T[i,j])/(dy**2) - 2*sigma*eps*(T[i,j]**4 - 273.15**4)/(k*0.02)

                        else:
                            fun[i,j] =(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + (T[i,j+1] + T[i,j-1] - 2*T[i,j])/(dy**2) - 2*sigma*eps*(T[i,j]**4 - 273.15**4)/(k*0.02)
       
        T = T.flatten()

        return fun.flatten()

     T6 = optimize.fsolve(f, T0)

     return  T6.reshape(N,Ny)

   # Caso - 6 : BIDIMENSIONAL X-Y ----------------------------------------------------

def T_case6xy(x, y, kx, ky, Q, sigma, eps, T3):

     N = len(x)
     Ny = len(y)

     dx = x[2]-x[1]
     dy = y[2]-y[1]
     q = Q/(0.1/dx)/(0.3*dx*dy)

     T0 = zeros([N, Ny])

     for j in range(0,Ny):
        T0[:,j] = T3
     T0 = T0.flatten()

     def f(T):
        fun = zeros([N,Ny])
        T = T.reshape(N,Ny)
        fun[0,:] = T[0,:] - 273.15
        fun[N-1,:] = T[N-1,:] - 273.15

        for i in range(1,N-1):
                for j in range(0,Ny):

                    if (x[i]>=0.05 and x[i]<=0.15) or (x[i]>=0.35 and x[i]<=0.45):
                        if y[j] == 0:
                            fun[i,j] = kx[0]*(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + ky[0]*(T[i,j+1] - T[i,j])/(dy**2)  - sigma*eps*(T[i,j]**4 - 273.15**4)/dy

                        elif y[j] == 0.02:
                            fun[i,j] = kx[0]*(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + ky[0]*(T[i,j-1] - T[i,j])/(dy**2) + q

                        elif (y[j]> 0 and y[j]<= 0.0025) or (y[j]>= 0.0175 and y[j]< 0.02):
                            fun[i,j] = kx[0]*(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + ky[0]*(T[i,j+1] + T[i,j-1] - 2*T[i,j])/(dy**2) 

                        else:
                            fun[i,j] = kx[1]*(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + ky[1]*(T[i,j+1] + T[i,j-1] - 2*T[i,j])/(dy**2) 

                    elif x[i]<0.05 or x[i]>0.45 or (x[i]>0.15 and x[i]<0.35):
                       if y[j] == 0:
                            fun[i,j] = kx[0]*(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) +  ky[0]*(T[i,j+1] - T[i,j])/(dy**2)  - sigma*eps*(T[i,j]**4 - 273.15**4)/dy

                       elif y[j] == 0.02:
                            fun[i,j] = kx[0]*(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + ky[0]*(T[i,j-1] - T[i,j])/(dy**2) - sigma*eps*(T[i,j]**4 - 273.15**4)/dy
                            
                       elif (y[j]> 0 and y[j]<= 0.0025) or (y[j]< 0.02 and y[j]>= 0.0175): 
                            fun[i,j] = kx[0]*(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + ky[0]*(T[i,j+1] + T[i,j-1] - 2*T[i,j])/(dy**2) 

                       else:
                            fun[i,j] = kx[1]*(T[i-1,j] + T[i+1,j] - 2*T[i,j])/(dx**2) + ky[1]*(T[i,j+1] + T[i,j-1] - 2*T[i,j])/(dy**2) 
       
        T = T.flatten()

        return fun.flatten()

     T6 = optimize.fsolve(f, T0)

     return  T6.reshape(N,Ny)

  # Caso - 6 : TRIDIMENSIONAL ----------------------------------------------------
 
def T_case7xyz(x, y, z, kx, ky, Q, sigma, eps, T3):

     kz = kx

     Nx = len(x)
     Ny = len(y)
     Nz = len(z)

     dx = x[2]-x[1]
     dy = y[2]-y[1]
     dz = z[2]-z[1]
     q = Q/(0.1*0.3/(dx*dz))/(dz*dx*dy)

     T0 = zeros([Nx, Ny, Nz])

     for i in range(0,Nz):
        for j in range(0,Ny):
            T0[:,j,i] = T3
     T0 = T0.flatten()

     def f(T):
        fun = zeros([Nx,Ny,Nz])
        T = T.reshape(Nx,Ny,Nz)
        fun[0,:,:] = T[0,:,:] - 273.15
        fun[Nx-1,:,:] = T[Nx-1,:] - 273.15

        for i in range(1,Nx-1):
                for j in range(0,Ny):
                    for n in range(0, Nz):

                            if n == 0:
                                if y[j] == 0:
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j+1,n] - T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n+1] -T[i,j,n])/(dz**2)  - sigma*eps*(T[i,j,n]**4 - 273.15**4)/dy

                                elif y[j] == 0.02:
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j-1,n] - T[i,j,n])/(dy**2) +  kz[0]*(T[i,j,n+1] -T[i,j,n])/(dz**2) - sigma*eps*(T[i,j,n]**4 - 273.15**4)/dy

                                elif (y[j]> 0 and y[j]<= 0.0025) or (y[j]>= 0.0175 and y[j]< 0.02):
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j+1,n] + T[i,j-1,n] - 2*T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n+1] -T[i,j,n])/(dz**2) 

                                else:
                                    fun[i,j,n] = kx[1]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[1]*(T[i,j+1,n] + T[i,j-1,n] - 2*T[i,j,n])/(dy**2) + kz[1]*( T[i,j,n+1] -T[i,j,n])/(dz**2) 
                                   
                            elif n == Nz-1:
                                if y[j] == 0:
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j+1,n] - T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n-1] -T[i,j,n])/(dz**2)  - sigma*eps*(T[i,j,n]**4 - 273.15**4)/dy

                                elif y[j] == 0.02:
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j-1,n] - T[i,j,n])/(dy**2) +  kz[0]*(T[i,j,n-1] -T[i,j,n])/(dz**2) - sigma*eps*(T[i,j,n]**4 - 273.15**4)/dy

                                elif (y[j]> 0 and y[j]<= 0.0025) or (y[j]>= 0.0175 and y[j]< 0.02):
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j+1,n] + T[i,j-1,n] - 2*T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n-1] -T[i,j,n])/(dz**2) 

                                else:
                                    fun[i,j,n] = kx[1]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[1]*(T[i,j+1,n] + T[i,j-1,n] - 2*T[i,j,n])/(dy**2) + kz[1]*( T[i,j,n-1] -T[i,j,n])/(dz**2) 
                                   

                            elif ((x[i]>=0.05 and x[i]<=0.15) and (z[n]>= 0.1 and z[n]<=0.2)) or ((x[i]>=0.35 and x[i]<=0.45) and (z[n]>= 0.1 and z[n]<=0.2)): # Heaters
                                if y[j] == 0:
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j+1,n] - T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n-1] + T[i,j,n+1] -2*T[i,j,n])/(dz**2)  - sigma*eps*(T[i,j,n]**4 - 273.15**4)/dy

                                elif y[j] == 0.02:
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j-1,n] - T[i,j,n])/(dy**2) +  kz[0]*(T[i,j,n-1] + T[i,j,n+1] -2*T[i,j,n])/(dz**2) + q

                                elif (y[j]> 0 and y[j]<= 0.0025) or (y[j]>= 0.0175 and y[j]< 0.02):
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j+1,n] + T[i,j-1,n] - 2*T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n-1] + T[i,j,n+1] -2*T[i,j,n])/(dz**2) 

                                else:
                                    fun[i,j,n] = kx[1]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[1]*(T[i,j+1,n] + T[i,j-1,n] - 2*T[i,j,n])/(dy**2) +  kz[1]*(T[i,j,n-1] + T[i,j,n+1] -2*T[i,j,n])/(dz**2) 

                            elif (x[i]<0.05) or (x[i]>0.45) or (x[i]>0.15 and x[i]<0.35) or ((x[i]>=0.05 and x[i]<=0.15) and (z[n]< 0.1)) or ((x[i]>=0.05 and x[i]<=0.15) and (z[n]>0.2)) or ((x[i]>=0.35 and x[i]<=0.45) and (z[n]< 0.1)) or ((x[i]>=0.35 and x[i]<=0.45) and (z[n]>0.2)):
                               if y[j] == 0:
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) +  ky[0]*(T[i,j+1,n] - T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n-1] + T[i,j,n+1] -2*T[i,j,n])/(dz**2)  - sigma*eps*(T[i,j,n]**4 - 273.15**4)/dy

                               elif y[j] == 0.02:
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j-1,n] - T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n-1] + T[i,j,n+1] -2*T[i,j,n])/(dz**2) - sigma*eps*(T[i,j,n]**4 - 273.15**4)/dy
                            
                               elif (y[j]> 0 and y[j]<= 0.0025) or (y[j]< 0.02 and y[j]>= 0.0175): 
                                    fun[i,j,n] = kx[0]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[0]*(T[i,j+1,n] + T[i,j-1,n] - 2*T[i,j,n])/(dy**2) + kz[0]*(T[i,j,n-1] + T[i,j,n+1] -2*T[i,j,n])/(dz**2) 

                               else:
                                    fun[i,j,n] = kx[1]*(T[i-1,j,n] + T[i+1,j,n] - 2*T[i,j,n])/(dx**2) + ky[1]*(T[i,j+1,n] + T[i,j-1,n] - 2*T[i,j,n])/(dy**2) + kz[1]*(T[i,j,n-1] + T[i,j,n+1] -2*T[i,j,n])/(dz**2) 
       
        T = T.flatten()

        return fun.flatten()

     T7 = optimize.fsolve(f, T0)

     return  T7.reshape(Nx,Ny, Nz)