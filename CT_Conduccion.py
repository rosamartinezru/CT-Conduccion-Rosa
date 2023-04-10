
from numpy import *
import Funciones as fn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from matplotlib import cm

# PARÁMETROS
N = 2000
x = linspace(0, 0.5, N+1)
y = linspace(0,0.02, 15)
z = linspace(0, 0.3, int(N/2)+1)
t = linspace(0, 1000, 1000)
A_tras = 0.3*0.02

Nx = len(x)
Ny = len(y)
Nz = len(z)
# MATERIALES
#Aluminio
Al_e = 0.015  # [m]
Al_k = 0.7 # [W/mK]
Al_rho = 40 # [kg/m^3]
Al_c = 890 # [J/kgK]

#CFRP
CFRP_e = 0.0025 # [m]
CFRP_k = 130 # [W/mK]
CFRP_rho = 1650 # [kg/m^3]
CFRP_c = 930 # [J/kgK]

# CONDUCTIVIDAD TÉRMICA EFECTIVA
Composite_k = array([CFRP_k, Al_k ,CFRP_k])
Composite_esp = array([CFRP_e, Al_e ,CFRP_e])
Composite_rho = array([CFRP_rho, Al_rho, CFRP_rho])
Composite_c = array([CFRP_c, Al_c, CFRP_c])

[k_eff_t, k_eff_lon] = fn.k_eff(Composite_k, Composite_esp); # [W/mK]

## CASO-1: DISIPACIÓN HOMOGÉNEA EN TODO EL PANEL -----------------------------------

Q1 = 100 # [W]
T_c1_an = fn.T_case1analytic(x, k_eff_lon, Q1); # [K]
T_c1 = fn.T_case1(x, k_eff_lon, Q1); # [K]

T_c1_max = max(T_c1)
T_c1_sensor = T_c1[int(N/2)]

T_c1_max_an = max(T_c1_an)
T_c1_sensor_an = T_c1_an[int(N/2)]


## CASO-2: HEATERS ------------------------------------------------------------

Q2 = 50 # [W]
T_c2_an = fn.T_case2analytic(x, k_eff_lon, Q2, A_tras);
T_c2 = fn.T_case2(x, k_eff_lon, Q2)

T_c2_max = max(T_c2)
T_c2_sensor = T_c2[int(N/2)]

T_c2_max_an = max(T_c2_an)
T_c2_sensor_an = T_c2_an[int(N/2)]

## CASO-3: PÉRDIDAS POR RADIACIÓN --------------------------------------------------

Sigma = 5.67e-8
Eps_CFRP = 0.8

h_rad = 4*Sigma*Eps_CFRP*T_c2_sensor**3

T_c3 = fn.T_case3(x, k_eff_lon, Q2, h_rad)
#T_c3_an = fn.T_case3analytic(x, k_eff_lon, Q2, A_tras, h_rad)
T_c3_nolineal = fn.T_case3_nolineal(x, k_eff_lon, Q2, Sigma, Eps_CFRP, T_c3)

T_c3_max = max(T_c3)
T_c3_sensor = T_c3[int(N/2)]

T_c3_max_nl = max(T_c3_nolineal)
T_c3_sensor_nl = T_c3_nolineal[int(N/2)]

## CASO-4 : EVOLUCIÓN TRANSITORIA UNIDIMENSIONAL --------------------------------

#Alpha = fn.alpha_eff(k_eff_lon, Composite_esp, Composite_rho, Composite_c) 
#T_c4 = fn.T_case4(t, x, k_eff_lon, Q2, Sigma, Eps_CFRP, Alpha, T_c3_nolineal)

## CASO-5 : TERMOSTATOS ----------------------------------------------------------

#T_c5 = fn.T_case5(t, x, k_eff_lon, Q2, Sigma, Eps_CFRP, Alpha, T_c3)

#T_c5sensor = T_c5[:,int(N/2)]
#T_c5_max = max(max(fila) for fila in T_c5)


## CASO-6 : MODELO BIDIMENSIONAL ESTACIONARIO ----------------------------------

#T_c6xz = fn.T_case6xz(x, z, k_eff_lon, Q2, Sigma, Eps_CFRP, T_c3)

#k_y = [1, 1.5]

#T_c6xy = fn.T_case6xy(x, y, Composite_k, k_y, Q2, Sigma, Eps_CFRP, T_c3)


## CASO-7 : MODELO TRIDIMENSIONAL --------------------------------------------

#T_c7xyz = fn.T_case7xyz(x, y, z, Composite_k, k_y, Q2, Sigma, Eps_CFRP, T_c3)

#T_c7_max = amax(T_c7xyz)


#####################################################################################

#plt.rcParams.update({
#    "text.usetex": True,
#    "font.family": "sans-serif",
#    "font.sans-serif": "avant",
#})

## GRÁFICA CASO ESTACIONARIO UNIDIMENSIONAL --------------------------------------------------

plt.plot(x, T_c1_an - 273.15,'k' , label = "Caso-1, Analitico")
plt.plot(x, T_c1 - 273.15, '--kd', label = "Caso-1, Numerico", markevery=100)
plt.plot(x, T_c2_an - 273.15, 'r', label = "Caso-2, Analitico")
plt.plot(x, T_c2 - 273.15, '--rd', label = "Caso-2, Numérico", markevery=100)
plt.plot(x, T_c3_nolineal - 273.15, '--sb', label = "Caso-3, Numérico sin linealizar", markevery=100)
plt.plot(x, T_c3 - 273.15, '--bd', label = "Caso-3, Numérico", markevery=100)
plt.title("Distribucion de temperaturas, modelo unidimensional estacionario")
plt.xlabel("x [m]")
plt.ylabel("T [$^\circ$C]")
plt.legend(loc = "upper left")
plt.grid()
plt.show()


## GRÁFICA EVOLUCIÓN TRANSITORIA UNIDIMENSIONAL --------------------------------------------

#leyenda = []
#for tt in range(0, len(t), 1):
#    leyenda.append(f"t = {round(t[tt],0)} s")
#    # Asignar un color diferente a cada curva
#    plt.plot(x, T_c4[tt,:]-273.15)
##leyenda.append("Caso estacionario")
## Asignar un color específico a la curva del caso estacionario
#plt.plot(x, T_c3_nolineal - 273.15, '--k', label = "Caso estacionario", markevery=100)
#plt.xlabel("x [m]")
#plt.ylabel("T [$^\circ$C]")
#plt.grid()
##plt.legend(leyenda, loc = "upper left")
#plt.title("Evolución de la distribución de temperaturas, modelo unidimensional")
#plt.show()


## GRÁFICA EVOLUCIÓN TRANSITORIA CON TERMOSTATOS UNIDIMENSIONAL ------------------------------

#leyenda = []
#for tt in range(0, len(t), 1):
#    leyenda.append(f"t = {round(t[tt],0)} s")
#    # Asignar un color diferente a cada curva
#    plt.plot(x, T_c5[tt,:]-273.15)
##leyenda.append("Caso estacionario")
#plt.plot(x, T_c3_nolineal - 273.15, '--k', label = "Caso estacionario", markevery=100)
#plt.xlabel("x [m]")
#plt.ylabel("T [$^\circ$C]")
#plt.grid()
##plt.legend(leyenda, loc = "upper left")
#plt.title("Evolución cíclica de la distribución de temperaturas, con termostatos")
#plt.show()

#leyenda = []
#for tt in range(0, int(N/2)+2, 50):
#    leyenda.append(f"x = {round(x[tt],4)} m")
#    # Asignar un color diferente a cada curva
#    plt.plot(t, T_c5[:,tt]-273.15)
#plt.xlabel("t [s]")
#plt.ylabel("T [$^\circ$C]")
#plt.grid()
#plt.legend(leyenda, loc = "upper left")
#plt.title("Evolución de la temperatura en diferentes puntos del panel, con termostatos")
#plt.show()


## GRÁFICA BIDIMENSIONAL --------------------------------------------------------------

#plt.pcolormesh(x, z, T_c6xz.transpose()-273.15, cmap='plasma')
#plt.xlabel('x [m]')
#plt.ylabel('z [m]')
#plt.title('Distribucion bidimensional de la temperatura en la cara x-z')
#plt.colorbar()  
#plt.show()

#plt.plot(x, T_c3_nolineal - 273.15, '--k', label = "1-D")
#plt.plot(x, T_c6xz[:,int(N/4)] - 273.15, 'k', label = "2-D")
#plt.xlabel("x [m]")
#plt.ylabel("T [$^\circ$C]")
#plt.grid()
#plt.legend(loc = "upper left")
#plt.title("Comparacion de la distribucion de temperatura para z = 0.15 m")
#plt.show()


#plt.pcolormesh(x, y, T_c6xy.transpose()-273.15, cmap='plasma')
#plt.xlabel('x [m]')
#plt.ylabel('y [m]')
#plt.title('Distribucion bidimensional de la temperatura en la cara transversal x-y')
#plt.colorbar()  
#plt.show()


#plt.plot(x, T_c3_nolineal - 273.15, '--k', label = "1-D")
#plt.plot(x, T_c6xz[:,int(N/4)] - 273.15, 'k', label = "2-D x-z")
#plt.plot(x, T_c6xy[:,0] - 273.15, 'b', label = "2-D x-y cara superior")
#plt.plot(x, T_c6xy[:,len(y)-1] - 273.15, 'r', label = "2-D x-y cara inferior")
#plt.xlabel("x [m]")
#plt.ylabel("T [$^\circ$C]")
#plt.grid()
#plt.legend(loc = "upper left")
#plt.title("Distribucion de temperatura en la cara superior e inferior del panel")
#plt.show()

## GRÁFICA TRIDIMENSIONAL ----------------------------------------------------------------

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#norm = colors.Normalize(vmin=T_c7xyz.flatten().min(), vmax=T_c7xyz.flatten().max())
#cmap = plt.cm.get_cmap('jet')

#for i in range(0,Nx):
#    for j in range(0,Ny):
#        for k in range(0,Nz):
#            ax.scatter(x[i], z[k], y[j], c=cmap(norm(T_c7xyz[i,j,k])), alpha=0.5)

#ax.view_init(elev=45, azim=45)

## Agregar etiquetas a los ejes
#ax.set_xlabel('x [m]')
#ax.set_ylabel('z [m]')
#ax.set_zlabel('y [m]')

## Crear el objeto de eje de color (cax)
#mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
#mappable.set_array([])
#fig.subplots_adjust(right=0.8)
#cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
#fig.colorbar(mappable, cax=cbar_ax, label='Temperatura [K]')

#plt.show()


#fig, axs = plt.subplots(2, 2)
#fig.suptitle('Distribucion de temperatura, cara x-z')
#cmap = 'plasma'
#norm = colors.Normalize(vmin=T_c7xyz.min()-273.15, vmax=T_c7xyz.max()-273.15)

#axs[0, 0].pcolormesh(x, z, T_c7xyz[:,len(y)-1,:].transpose()-273.15, cmap=cmap, norm=norm)
#axs[0, 0].set_title(f"y = {y[len(y)-1]} m")
#axs[0, 1].pcolormesh(x, z, T_c7xyz[:,round(int(2*Ny/3),0),:].transpose()-273.15, cmap=cmap, norm=norm)
#axs[0, 1].set_title(f"y = {round(y[round(int(2*Ny/3),0)],4)} m")
#axs[1, 0].pcolormesh(x, z, T_c7xyz[:,round(int(Ny/3),0),:].transpose()-273.15, cmap=cmap, norm=norm)
#axs[1, 0].set_title(f"y = {round(y[round(int(Ny/3),0)],4)} m")
#axs[1, 1].pcolormesh(x, z, T_c7xyz[:,0,:].transpose()-273.15, cmap=cmap, norm=norm)
#axs[1, 1].set_title(f"y = {y[0]} m")

#for ax in axs.flat:
#    ax.set(xlabel='x [m]', ylabel='z [m]')

#fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs.ravel().tolist())

#plt.show()


#fig, axs = plt.subplots(2, 2)
#fig.suptitle('Distribución de temperatura, cara x-y')
#cmap = 'plasma'
#norm = colors.Normalize(vmin=T_c7xyz.min()-273.15, vmax=T_c7xyz.max()-273.15)

#axs[0, 0].pcolormesh(x, y, T_c7xyz[:,:,len(z)-1].transpose()-273.15, cmap=cmap, norm=norm)
#axs[0, 0].set_title(f"z = {z[len(z)-1]} m")
#axs[0, 1].pcolormesh(x, y, T_c7xyz[:,:,round(int(2*Nz/3),0)].transpose()-273.15, cmap=cmap, norm=norm)
#axs[0, 1].set_title(f"z = {round(z[round(int(2*Nz/3),0)],4)} m")
#axs[1, 0].pcolormesh(x, y, T_c7xyz[:,:,round(int(Nz/3),0)].transpose()-273.15, cmap=cmap, norm=norm)
#axs[1, 0].set_title(f"z = {round(z[round(int(Nz/3),0)],4)} m")
#axs[1, 1].pcolormesh(x, y, T_c7xyz[:,:,0].transpose()-273.15, cmap=cmap, norm=norm)
#axs[1, 1].set_title(f"z = {z[0]} m")

#for ax in axs.flat:
#    ax.set(xlabel='x [m]', ylabel='y [m]')

#fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs.ravel().tolist())

#plt.show()

#fig, axs = plt.subplots(2, 2)
#fig.suptitle('Distribución de temperatura, cara y-z')
#cmap = 'plasma'
#norm = colors.Normalize(vmin=T_c7xyz.min()-273.15, vmax=T_c7xyz.max()-273.15)

#axs[0, 0].pcolormesh(z, y, T_c7xyz[len(x)-1,:,:]-273.15, cmap=cmap, norm=norm)
#axs[0, 0].set_title(f"x = {x[len(x)-1]} m")
#axs[0, 1].pcolormesh(z, y, T_c7xyz[round(int(2*Nx/3),0),:,:]-273.15, cmap=cmap, norm=norm)
#axs[0, 1].set_title(f"x = {round(x[round(int(2*Nx/3),0)],4)} m")
#axs[1, 0].pcolormesh(z, y, T_c7xyz[round(int(Nx/3),0),:,:]-273.15, cmap=cmap, norm=norm)
#axs[1, 0].set_title(f"x = {round(x[round(int(Nx/3),0)],4)} m")
#axs[1, 1].pcolormesh(z, y, T_c7xyz[0,:,:]-273.15, cmap=cmap, norm=norm)
#axs[1, 1].set_title(f"x = {x[0]} m")

#for ax in axs.flat:
#    ax.set(xlabel='z [m]', ylabel='y [m]')

#fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axs.ravel().tolist())

#plt.show()