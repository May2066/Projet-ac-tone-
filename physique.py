"""Noyau physique du mini-projet : propagation, interférences, capteur CuO, dosage.

Toutes les phases sont en RADIANS dans ce module (conversion en degrés dans l'interface).
Origine des données : Rydosz, Brudnik, Staszek, Materials 12(6), 877 (2019).
"""
import numpy as np

F_REF_HZ = 2.4e9          # fréquence expérimentale publiée
C_MAX_PPM = 200.0         # plage de l'étude citée
PHI_200_DEG = 3.0         # variation de phase S21 publiée à 200 ppm (≈)
BRUIT_PHASE_DEG = 0.04    # bruit de phase rapporté (≈)


# ---------- A. Propagation avec pertes ----------
def onde(z, wt, alpha, beta, E0=1.0):
    """E(z,t) = E0 exp(-alpha z) cos(wt - beta z).  alpha en Np/m, beta en rad/m, wt en rad."""
    return E0 * np.exp(-alpha * z) * np.cos(wt - beta * z)


def profondeur_penetration(alpha):
    """1/alpha (m) ; infinie si alpha = 0."""
    return np.inf if alpha <= 0 else 1.0 / alpha


# ---------- B/C. Interféromètre idéal (coupleurs 3 dB) ----------
def puissance(ar, am, dphi):
    """Ps/P0 = 1/4 |hr + hm|^2 = 1/4 [ar^2 + am^2 + 2 ar am cos(dphi)]."""
    return 0.25 * (ar**2 + am**2 + 2 * ar * am * np.cos(dphi))


def puissance_complexe(ar, am, dphi):
    """Même résultat calculé directement avec les nombres complexes (vérification croisée)."""
    return 0.25 * np.abs(ar + am * np.exp(1j * dphi)) ** 2


def contraste(ar, am):
    """V = (Pmax - Pmin)/(Pmax + Pmin) = 2 ar am / (ar^2 + am^2)."""
    return 2 * ar * am / (ar**2 + am**2)


def sensibilite_phase(ar, am, dphi):
    """S_phi = |dPs/P0 / d(dphi)| = (ar am / 2) |sin(dphi)|."""
    return 0.5 * ar * am * np.abs(np.sin(dphi))


# ---------- D. Capteur CuO ----------
def dphi_cuo(C, mode, pts=None, phi_max_deg=PHI_200_DEG, n=1.0, signe=+1):
    """Variation de phase de la voie sensible (rad) en fonction de la concentration C (ppm).

    mode 'experimental' : interpolation LINÉAIRE entre les points documentés `pts` = [(C, phi_deg), ...],
                          uniquement dans leur domaine (NaN en dehors).
    mode 'parametrique' : phi_max * (C/200)^n  -> HYPOTHÈSE, pas une mesure.
    """
    C = np.asarray(C, dtype=float)
    if mode == "experimental":
        pts = np.array(sorted(pts), dtype=float)
        phi = np.interp(C, pts[:, 0], pts[:, 1], left=np.nan, right=np.nan)
    else:
        phi = phi_max_deg * (np.clip(C, 0, None) / C_MAX_PPM) ** n
    return signe * np.deg2rad(phi)


def reponse_P(C, ar, am, biais, **kw):
    """Ps(C)/P0 avec dphi_total = dphi_CuO(C) + biais."""
    return puissance(ar, am, dphi_cuo(C, **kw) + biais)


# ---------- E. Dosage inverse ----------
def inverser(P_mesure, grille_C, P_grille):
    """Retourne toutes les concentrations compatibles avec P_mesure (croisements de la courbe P(C)).

    Plusieurs solutions => inversion NON unique sur la plage."""
    d = P_grille - P_mesure
    ok = ~np.isnan(d)
    sols = []
    for i in range(len(grille_C) - 1):
        if ok[i] and ok[i + 1] and d[i] * d[i + 1] <= 0 and d[i] != d[i + 1]:
            t = d[i] / (d[i] - d[i + 1])
            sols.append(grille_C[i] + t * (grille_C[i + 1] - grille_C[i]))
    return np.unique(np.round(sols, 6))


def est_monotone(y):
    y = y[~np.isnan(y)]
    dy = np.diff(y)
    return bool(np.all(dy >= 0) or np.all(dy <= 0))


def resolution_ppm(ar, am, biais, sigma_phi_rad, C0, h=0.5, **kw):
    """Incertitude sur C : sigma_C ≈ sigma_P / |dP/dC|, avec sigma_P = S_phi * sigma_phi."""
    d = dphi_cuo(C0, **kw) + biais
    dP_dC = (reponse_P(C0 + h, ar, am, biais, **kw) - reponse_P(C0 - h, ar, am, biais, **kw)) / (2 * h)
    if not np.isfinite(dP_dC) or abs(dP_dC) < 1e-15:
        return np.inf
    return sensibilite_phase(ar, am, d) * sigma_phi_rad / abs(dP_dC)


if __name__ == "__main__":
    # Auto-tests rapides
    ph = np.linspace(0, 2 * np.pi, 3601)
    for ar, am in [(1, 1), (1, .8), (1, .5)]:
        P = puissance(ar, am, ph)
        assert np.allclose(P, puissance_complexe(ar, am, ph))
        assert P.min() >= 0 and P.max() <= 1 + 1e-12
        V = (P.max() - P.min()) / (P.max() + P.min())
        assert abs(V - contraste(ar, am)) < 1e-6
        i = np.argmax(sensibilite_phase(ar, am, ph[:1801]))
        assert abs(np.rad2deg(ph[i]) - 90) < 0.1
        print(f"ar/am={ar}/{am}  V={V:.4f}  Pmax={P.max():.3f} Pmin={P.min():.3f}  OK")
    kw = dict(mode="parametrique", n=1.0)
    C = np.linspace(0, 200, 2001)
    P = reponse_P(C, 1, 1, np.pi / 2, **kw)
    assert abs(dphi_cuo(200, **kw) - np.deg2rad(3)) < 1e-12
    print("monotone sur 0-200 ppm (quadrature) :", est_monotone(P))
    s = resolution_ppm(1, 1, np.pi / 2, np.deg2rad(BRUIT_PHASE_DEG), 100, **kw)
    print(f"résolution (hyp. linéaire, quadrature) ≈ {s:.2f} ppm")
    sols = inverser(reponse_P(80.0, 1, 1, np.pi / 2, **kw), C, P)
    print("inversion de 80 ppm ->", sols)
    print("Tous les tests passent.")
