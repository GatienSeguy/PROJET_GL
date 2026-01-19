import numpy as np

def amplitude_geste(p0, p1, dx):
    p0 = np.array(p0)
    p1 = np.array(p1)

    distance = np.linalg.norm(p1 - p0) * dx
    return distance


def lucas_kanade(Ix, Iy, It):
    A = np.column_stack((Ix, Iy))
    P = -It

    v, _, _, _ = np.linalg.lstsq(A, P, rcond=None)

    return v


n = 5
N = n * n


Ix = np.random.randn(N)
Iy = np.random.randn(N)
It = np.random.randn(N)

v = lucas_kanade(Ix, Iy, It)

print("Vecteur vitesse estimé :", v)



import numpy as np
import matplotlib.pyplot as plt




N, M = 80, 80          
dx = dy = 1.0          
delta_t = 1 / 11      

x = np.linspace(0, M-1, M)
y = np.linspace(0, N-1, N)
X, Y = np.meshgrid(x, y)


x0, y0 = 40, 40
sigma = 6
I_t = np.exp(-((X-x0)**2 + (Y-y0)**2) / (2*sigma**2))


vx_true, vy_true = 1.5, -0.8
I_t_dt = np.exp(-((X-(x0+vx_true))**2 + (Y-(y0+vy_true))**2) / (2*sigma**2))



Ix = np.gradient(I_t, axis=1) / dx
Iy = np.gradient(I_t, axis=0) / dy
It = (I_t_dt - I_t) / delta_t



w = 15
i0, j0 = int(y0), int(x0)

Ix_w = Ix[i0-w:i0+w, j0-w:j0+w].ravel()
Iy_w = Iy[i0-w:i0+w, j0-w:j0+w].ravel()
It_w = It[i0-w:i0+w, j0-w:j0+w].ravel()



A = np.column_stack((Ix_w, Iy_w))
P = -It_w
v_est, _, _, _ = np.linalg.lstsq(A, P, rcond=None)



plt.figure()
plt.imshow(I_t, cmap='gray')
plt.title("Image IR à l'instant t")
plt.colorbar()

plt.figure()
plt.imshow(I_t_dt, cmap='gray')
plt.title("Image IR à l'instant t + Δt")
plt.colorbar()

plt.figure()
plt.imshow(I_t_dt - I_t, cmap='gray')
plt.title("Différence temporelle I(t+Δt) - I(t)")
plt.colorbar()

plt.figure()
plt.imshow(I_t, cmap='gray')
plt.quiver(
    x0, y0,
    v_est[0], v_est[1],
    color='red',
    scale=10
)
plt.scatter([x0], [y0], color='blue', label="Centre main")
plt.legend()
plt.title("Vecteur vitesse estimé (Lucas–Kanade)")

plt.show()


v_est, (vx_true, vy_true)