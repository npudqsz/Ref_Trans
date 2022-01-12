# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:05:52 2021

@author: qdu

objective : calculate reflection & refraction angle & coefficients
            in function of inputs

paper : 'Reflection and Transmission of Oblique Plane Waves at 
a Plane Interface between Viscoelastic Media' by H. F. Cooper Jr, 1967

characteristics : small amplitude time harmonic plane waves
in homogeneous and isotropic linearly viscoelastic materials

notation :
l = 0 : incident P-wave
l = 1 : incident S-wave
m = 0 : y > 0 zone (incident side)
m = 1 : y < 0 zone
n = 0 : reflected/refracted P-wave
n = 1 : reflected/refracted S-wave

"""
import numpy as np
from numpy import sin
from numpy import cos
from numpy import tan
from numpy import tanh
from numpy import arcsin
from numpy import arctan
from numpy import arcsinh

def E_complex(E_inf, Ev, eta, omega):
    """ complex Young modulus of one-branch GM model """
    tau = eta / Ev
    Ed = E_inf + (tau*omega)**2 / (1 + (tau*omega)**2) * Ev
    El = tau*omega / (1 + (tau*omega)**2) * Ev
    return complex(Ed, El)

def angle_coef(thetaI, E_inf, Ev, eta, rho=[1000,1000], nu=[0.49,0.49], f=100):

    omega = 2*np.pi*f

    # complex Young's modulus
    E = np.zeros(2, dtype=complex)
    for m in range(2):
        E[m] = E_complex(E_inf[m], Ev[m], eta[m], omega)

    # complex Lame parameters
    lamb = np.zeros(2, dtype=complex)
    mu = np.zeros(2, dtype=complex)
    for m in range(2):
        lamb[m] = E[m] * nu[m] / (1+nu[m]) / (1-2*nu[m])
        mu[m] = E[m] / 2 / (1+nu[m])

    # Smn²
    # Sm0² = (lamb_m + 2*mu_m) / rho_m
    # Sm1² = mu_m / rho_m
    S_sq = np.zeros((2,2), dtype=complex)
    for m in range(2):
        S_sq[m,0] = (lamb[m] + 2*mu[m]) / rho[m]
        S_sq[m,1] = mu[m] / rho[m]

    # Omega_mn
    # tan(2*Omega_mn) = + Im(Smn²) / Re(Smn²)
    # => modified from minus to plus, requiring that in other equations, Omega[m,n] should be modified to -Omega[m,n]
    # 0 <= Omega_mn < pi/2
    Omega = np.zeros((2,2))
    for m in range(2):
        for n in range(2):
            tmp = + S_sq[m,n].imag / S_sq[m,n].real     # plus instead of minus
            if tmp < 0:
                Omega[m,n] = (arctan(tmp) + np.pi) / 2
            else:
                Omega[m,n] = arctan(tmp) / 2
            
            if Omega[m,n] < 0 or Omega[m,n] >= np.pi/2:
                print("error : Omega -> ", Omega[m,n])

    # Cmn = |Smn| * sec(Omega_mn)
    C = np.zeros((2,2))
    for m in range(2):
        for n in range(2):
            C[m,n] = abs(np.sqrt(S_sq[m,n])) / cos(Omega[m,n])
            
    # gamma_lmn = Cmn * sin(theta_l) / C_0l
    gamma = np.zeros((2,2,2))
    for l in range(2):
        for m in range(2):
            for n in range(2):
                gamma[l,m,n] = C[m,n] * sin(thetaI[l]) / C[0,l]
                
    # Gamma_lmn = gamma_lmn * cos(Omega_mn) / cos(Omega_0l)
    Gamma = np.zeros((2,2,2))
    for l in range(2):
        for m in range(2):
            for n in range(2):
                Gamma[l,m,n] = gamma[l,m,n] * cos(Omega[m,n]) / cos(Omega[0,l])
                
    # Delta_lmn = Omega_0l - Omega_mn
    Delta = np.zeros((2,2,2))
    for l in range(2):
        for m in range(2):
            for n in range(2):
                Delta[l,m,n] = -Omega[0,l] - -Omega[m,n]
                
    # sinh²(beta_lmn) = 0.5 * { Gamma_lmn² - 1 + [(1-Gamma_lmn²)² + 4*Gamma_lmn²*sin²(Delta_lmn)]^0.5 }
    sinh_sq_beta = np.zeros((2,2,2))
    for l in range(2):
        for m in range(2):
            for n in range(2):
                sinh_sq_beta[l,m,n] = 0.5 * ( Gamma[l,m,n]**2 - 1
                                                 + ( (1-Gamma[l,m,n]**2)**2 + 
                                                        4*Gamma[l,m,n]**2 * sin(Delta[l,m,n])**2
                                                    )**0.5
                                             )
                
    # xi_lmn = (1 + sinh²(beta_lmn) * sec²(Omega_mn))^(-0.5)
    xi = np.zeros((2,2,2))
    for l in range(2):
        for m in range(2):
            for n in range(2):
                xi[l,m,n] = (1 + sinh_sq_beta[l,m,n] / cos(Omega[m,n])**2 )**(-0.5)
                
    # real angle theta
    # sin(theta_lmn) = xi_lmn * gamma_lmn
    theta = np.zeros((2,2,2))
    for l in range(2):
        for m in range(2):
            for n in range(2):
                theta[l,m,n] = arcsin(xi[l,m,n] * gamma[l,m,n])
                
    # complex angle zeta
    # zeta_lmn = alpha_lmn + i beta_lmn
    # beta_lmn :
    beta = np.zeros((2,2,2))
    for l in range(2):
        for m in range(2):
            for n in range(2):
                sgn = - np.sign(xi[l,m,n]**2 * tan(-Omega[m,n]) - tan(-Omega[0,l]))
                beta[l,m,n] = sgn * arcsinh(abs(np.sqrt(sinh_sq_beta[l,m,n])))
    # alpha_lmn & zeta_lmn :
    alpha = np.zeros((2,2,2))
    zeta = np.zeros((2,2,2), dtype=complex)
    for l in range(2):
        for m in range(2):
            for n in range(2):
                alpha[l,m,n] = theta[l,m,n] + arctan(tan(-Omega[m,n]) * tanh(beta[l,m,n]))
                zeta[l,m,n] = complex(alpha[l,m,n], beta[l,m,n])
                
    # S_mn
    S = np.zeros((2,2), dtype=complex)
    for m in range(2):
        for n in range(2):
            S[m,n] = np.sqrt(S_sq[m,n])
                        
    # solve coefficients R_lmn
    # A_l * R_l = B_l
    # R_l = {R_l00, R_l01, R_l10, R_l11}
    A = np.zeros((2,4,4), dtype=complex)
    B = np.zeros((2,4,1), dtype=complex)
    R = np.zeros((2,4,1), dtype=complex)
    for l in range(2):
        A[l,0,:] = np.array([sin(zeta[l,0,0]), cos(zeta[l,0,1]), -sin(zeta[l,1,0]), cos(zeta[l,1,1])])
        A[l,1,:] = np.array([cos(zeta[l,0,0]), -sin(zeta[l,0,1]), cos(zeta[l,1,0]), sin(zeta[l,1,1])])
        A[l,2,:] = np.array([-rho[0]*S[0,0]*cos(2*zeta[l,0,1]), rho[0]*S[0,1]*sin(2*zeta[l,0,1]),
                             rho[1]*S[1,0]*cos(2*zeta[l,1,1]), rho[1]*S[1,1]*sin(2*zeta[l,1,1])])
        A[l,3,:] = np.array([rho[0]*S_sq[0,1]/S[0,0]*sin(2*zeta[l,0,0]), rho[0]*S[0,1]*cos(2*zeta[l,0,1]),
                             rho[1]*S_sq[1,1]/S[1,0]*sin(2*zeta[l,1,0]), -rho[1]*S[1,1]*cos(2*zeta[l,1,1])])
    B[0,:,0] = np.array([-sin(thetaI[0]), cos(thetaI[0]),
                         rho[0]*S[0,0]*cos(2*zeta[0,0,1]), rho[0]*S_sq[0,1]/S[0,0]*sin(2*thetaI[0])])
    B[1,:,0] = np.array([cos(thetaI[1]), sin(thetaI[1]),
                         +rho[0]*S[0,1]*sin(2*thetaI[1]), -rho[0]*S[0,1]*cos(2*thetaI[1])])

    R[0,:,0] = np.linalg.solve(A[0,:,:], B[0,:,0])
    R[1,:,0] = np.linalg.solve(A[1,:,:], B[1,:,0])

    return theta, R
    
if __name__ == '__main__':
    
    # amplitude of incident P-wave (µm)
    A_i = 150
    
    # angle of incidence of P-wave (degree)
    alpha_i = 40
    
    thetaI=(alpha_i*np.pi/180, 0)
    E_inf = (6e3, 20e3)
    Ev=(10e3, 10e3)
    eta=(16, 16)
    rho=[1000, 1000]
    nu=[0.49, 0.49]
    f=100

    theta, R = angle_coef(thetaI, E_inf, Ev, eta, rho, nu, f)
    theta_degree = theta * 180/np.pi
    R_abs = abs(R)
    Amps = A_i * R_abs
    
    print('theta (degree):')
    print(theta_degree)
    print('')
    print('R:')
    print(R_abs)
    print('')
    print('Amplitude (µm):')
    print(Amps)