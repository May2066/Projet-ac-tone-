"""Application interactive : propagation et interférences micro-ondes, détection d'acétone (CuO).
Lancement :  streamlit run app.py
"""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
import physique as ph

st.set_page_config(page_title="Interféromètre micro-ondes – acétone", layout="wide")
st.title("Propagation et interférences micro-ondes : détection d'acétone")

# ---------------- Barre latérale : régime et paramètres globaux ----------------
sb = st.sidebar
if sb.button("Remise à zéro"):
    st.session_state.clear()
    st.rerun()

sb.header("Régime de données (module D/E)")
regime = sb.radio(
    "Régime", ["Expérimental 2019 (points documentés)", "Paramétrique (hypothèse)", "Idéal (sans capteur)"], key="regime"
)
sb.header("Interféromètre")
ar = sb.slider("Amplitude voie référence aᵣ", 0.1, 1.0, 1.0, 0.05, key="ar")
am = sb.slider("Amplitude voie mesure aₘ", 0.1, 1.0, 1.0, 0.05, key="am")
biais = np.deg2rad(sb.slider("Biais de phase φ_b (°)", 0, 360, 90, 1, key="biais"))

kw = {}
if regime.startswith("Expérimental"):
    sb.caption("Points (C, Δφ) issus de Rydosz et al. 2019. Complète-les seulement avec des valeurs lues sur les figures.")
    df = sb.data_editor(
        pd.DataFrame({"C (ppm)": [0.0, 200.0], "Δφ (°)": [0.0, ph.PHI_200_DEG]}),
        num_rows="dynamic", key="pts",
    )
    pts = [(float(a), float(b)) for a, b in df.dropna().values]
    kw = dict(mode="experimental", pts=pts)
    if len(pts) <= 2:
        sb.warning("Seulement 2 points : l'interpolation linéaire entre eux est une convention, pas une mesure.")
elif regime.startswith("Paramétrique"):
    n = sb.slider("Exposant n de Δφ = 3°·(C/200)ⁿ", 0.3, 2.0, 1.0, 0.1, key="n")
    kw = dict(mode="parametrique", n=n)
else:
    kw = dict(mode="parametrique", phi_max_deg=0.0)
signe = +1 if sb.radio("Signe de Δφ_CuO", ["+", "−"], horizontal=True, key="signe") == "+" else -1
kw["signe"] = signe
sb.info(f"Régime actif : **{regime}**")

tabs = st.tabs(["A Propagation", "B Interféromètre", "C Interférences", "D Capteur CuO", "E Dosage", "F Analyse", "G Vue 3D"])

# ---------------- A ----------------
with tabs[0]:
    st.subheader("Onde progressive dans un milieu avec pertes")
    c1, c2, c3, c4 = st.columns(4)
    alpha = c1.slider("α (Np/m)", 0.0, 10.0, 2.0, 0.1)
    beta = c2.slider("β (rad/m)", 1.0, 60.0, 20.0, 1.0)
    L = c3.slider("L (m)", 0.2, 5.0, 2.0, 0.1)
    wt = np.deg2rad(c4.slider("ωt (°)", 0, 720, 0, 5))
    z = np.linspace(0, L, 1500)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(z, ph.onde(z, wt, alpha, beta), label="E(z,t)")
    ax.plot(z, np.exp(-alpha * z), "k--", lw=1, label="enveloppe ±e^(−αz)")
    ax.plot(z, -np.exp(-alpha * z), "k--", lw=1)
    ax.set_xlabel("z (m)"); ax.set_ylabel("E / E₀"); ax.legend(loc="upper right"); ax.grid(alpha=.3)
    st.pyplot(fig)
    lam = 2 * np.pi / beta
    d = ph.profondeur_penetration(alpha)
    st.write(
        f"Longueur d'onde guidée λ = 2π/β = **{lam:.4f} m** · profondeur de pénétration 1/α = "
        f"**{'∞' if np.isinf(d) else f'{d:.3f} m'}** · déphasage βL = **{np.rad2deg(beta*L):.1f}°** · "
        f"atténuation e^(−αL) = **{np.exp(-alpha*L):.4f}** ({-20*np.log10(np.exp(-alpha*L)+1e-300):.2f} dB)"
    )
    st.caption("Continuité tangentielle : n×(E₂−E₁)=0 et n×(H₂−H₁)=0 aux interfaces. "
               "Un coefficient de Fresnel de surface plane ne donne PAS la phase S₂₁ d'une ligne guidée à couche CuO.")

# ---------------- B ----------------
with tabs[1]:
    st.subheader("Deux voies, somme cohérente")
    dphi_b = st.slider("Δφ entre les deux voies (°)", 0, 360, int(np.rad2deg(biais)), 1, key="dphi_b")
    x = np.linspace(0, 4 * np.pi, 800)
    er = ar * np.cos(x); em = am * np.cos(x + np.deg2rad(dphi_b)); es = 0.5 * (er + em)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(x, er, label="voie référence"); ax.plot(x, em, label="voie mesure"); ax.plot(x, es, "k", lw=2, label="sortie (somme/2)")
    ax.set_xlabel("ωt (rad)"); ax.set_ylabel("champ (u.a.)"); ax.legend(loc="upper right"); ax.grid(alpha=.3)
    st.pyplot(fig)
    P = ph.puissance(ar, am, np.deg2rad(dphi_b))
    st.write(f"Ps/P₀ = **{P:.4f}** (formule) · **{ph.puissance_complexe(ar, am, np.deg2rad(dphi_b)):.4f}** (complexe) · "
             f"contraste V = **{ph.contraste(ar, am):.4f}**")

# ---------------- C ----------------
with tabs[2]:
    st.subheader("Ps/P₀ en fonction de la phase")
    phi = np.linspace(0, 2 * np.pi, 721)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    rows = []
    for r in (1.0, 0.8, 0.5):
        P = ph.puissance(1.0, r, phi)
        ax.plot(np.rad2deg(phi), P, label=f"aᵣ/aₘ = {1/r:.2f}" if r != 1 else "aᵣ/aₘ = 1")
        rows.append({"aₘ (aᵣ=1)": r, "Pmax": P.max(), "Pmin": P.min(),
                     "V numérique": (P.max()-P.min())/(P.max()+P.min()), "V formule": ph.contraste(1.0, r)})
    ax.set_xlabel("Δφ (°)"); ax.set_ylabel("Ps/P₀"); ax.set_ylim(-0.02, 1.02); ax.legend(); ax.grid(alpha=.3)
    st.pyplot(fig)
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True)
    st.markdown("**Comparaison des réglages** (aᵣ, aₘ courants, petite variation de phase de 3°)")
    comp = []
    for b in (0, 90, 180):
        P1, P2 = ph.puissance(ar, am, np.deg2rad(b)), ph.puissance(ar, am, np.deg2rad(b + 3))
        comp.append({"biais (°)": b, "Ps/P₀": P1, "ΔP pour +3°": P2 - P1, "S_φ (par rad)": ph.sensibilite_phase(ar, am, np.deg2rad(b))})
    st.dataframe(pd.DataFrame(comp).round(5), hide_index=True)
    st.caption("Un maximum de puissance (0° ou 180°) a une sensibilité nulle : ce n'est pas le meilleur point de mesure.")

# ---------------- D ----------------
Cg = np.linspace(0, ph.C_MAX_PPM, 801)
with tabs[3]:
    st.subheader("Réponse du capteur CuO")
    dphi_c = ph.dphi_cuo(Cg, **kw)
    Pc = ph.puissance(ar, am, dphi_c + biais)
    S_loc = np.abs(np.gradient(Pc, Cg))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.4))
    axs[0].plot(Cg, np.rad2deg(dphi_c)); axs[0].set_title("Δφ_CuO(C)"); axs[0].set_ylabel("°")
    for b in (0, 90, 180):
        axs[1].plot(Cg, ph.puissance(ar, am, dphi_c + np.deg2rad(b)), lw=1, alpha=.6, label=f"biais {b}°")
    axs[1].plot(Cg, Pc, "k", lw=2, label="biais choisi"); axs[1].set_title("Ps(C)/P₀"); axs[1].legend(fontsize=7)
    axs[2].plot(Cg, S_loc); axs[2].set_title("|dP/dC| (par ppm)")
    for a in axs: a.set_xlabel("C (ppm)"); a.grid(alpha=.3)
    st.pyplot(fig)
    st.write(f"Test de cohérence : Δφ(200 ppm) = **{np.rad2deg(dphi_c[-1]):.3f}°** (cible publiée ≈ {ph.PHI_200_DEG}°) · "
             f"ΔP total 0→200 ppm = **{Pc[-1]-Pc[0]:+.5f}**")
    if regime.startswith("Paramétrique"):
        st.warning("Mode PARAMÉTRIQUE : la forme de la courbe est une hypothèse, seule la valeur à 200 ppm est publiée.")
    d1 = pd.DataFrame({"C (ppm)": Cg, "dphi (deg)": np.rad2deg(dphi_c), "Ps/P0": Pc, "regime": regime})
    st.download_button("Exporter la courbe (CSV)", d1.to_csv(index=False).encode(), "courbe_capteur.csv")

# ---------------- E ----------------
with tabs[4]:
    st.subheader("Dosage inverse sur un échantillon numérique")
    sigma_deg = st.slider("Bruit de phase σ (°)", 0.0, 0.5, ph.BRUIT_PHASE_DEG, 0.01)
    c_true = st.slider("Concentration « inconnue » générée (ppm)", 0.0, 200.0, 120.0, 1.0)
    rng = np.random.default_rng(st.number_input("Graine aléatoire", 0, 9999, 1))
    mono = ph.est_monotone(Pc)
    st.write("P(C) monotone sur la plage : " + ("**oui** → inversion unique." if mono else "**non** → inversion ambiguë."))
    d_true = ph.dphi_cuo(c_true, **kw) + rng.normal(0, np.deg2rad(sigma_deg)) + biais
    P_meas = float(ph.puissance(ar, am, d_true))
    sols = ph.inverser(P_meas, Cg, Pc)
    s_c = ph.resolution_ppm(ar, am, biais, np.deg2rad(sigma_deg), max(c_true, 1.0), **kw)
    st.write(f"P mesurée = **{P_meas:.6f}** · solutions : **{', '.join(f'{s:.1f}' for s in sols) if len(sols) else 'aucune'} ppm** · "
             f"valeur vraie = {c_true:.1f} ppm · incertitude estimée σ_C ≈ **{s_c:.2f} ppm**")
    st.caption("σ_C = S_φ·σ_φ / |dP/dC|. Valable seulement si la forme de Δφ(C) est correcte : "
               "avec un seul point publié, la précision réelle reste indéterminée.")
    bs = np.deg2rad(np.arange(10, 351, 10))
    sc = [ph.resolution_ppm(ar, am, b, np.deg2rad(sigma_deg), 100.0, **kw) for b in bs]
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.semilogy(np.rad2deg(bs), sc); ax.set_xlabel("biais (°)"); ax.set_ylabel("σ_C à 100 ppm"); ax.grid(alpha=.3)
    st.pyplot(fig)

# ---------------- F ----------------
with tabs[5]:
    st.subheader("Contrôles de cohérence et hypothèses")
    phis = np.linspace(0, 2 * np.pi, 3601)
    Pall = ph.puissance(ar, am, phis)
    Vs = []
    for dp in (0, 1, 3, 10, 45):
        P = ph.puissance(ar, am, phis + np.deg2rad(dp))
        Vs.append((P.max() - P.min()) / (P.max() + P.min()))
    iS = np.argmax(ph.sensibilite_phase(ar, am, phis[:1801]))
    checks = [
        ("0 ≤ Ps/P₀ ≤ 1", bool(Pall.min() >= -1e-12 and Pall.max() <= 1 + 1e-12)),
        ("Formule = calcul complexe", bool(np.allclose(Pall, ph.puissance_complexe(ar, am, phis)))),
        ("V indépendant d'un déphasage ajouté (amplitudes fixes)", bool(np.ptp(Vs) < 1e-9)),
        (f"V numérique = formule ({ph.contraste(ar, am):.4f})", abs(Vs[0] - ph.contraste(ar, am)) < 1e-6),
        (f"Sensibilité maximale à 90° (trouvée : {np.rad2deg(phis[iS]):.1f}°)", abs(np.rad2deg(phis[iS]) - 90) < 0.2),
    ]
    st.table(pd.DataFrame(checks, columns=["Contrôle", "OK"]).assign(OK=lambda d: d["OK"].map({True: "✔", False: "✘"})))
    st.markdown(
        "**Hypothèses affichées** : coupleurs idéaux 3 dB · deux voies cohérentes · amplitudes constantes avec C · "
        "température 23 °C et humidité 50 % supposées fixes · gaz unique (acétone).\n\n"
        "**Origine des données** : Δφ ≈ 3° à 200 ppm, 2,4 GHz, bruit de phase ≈ 0,04° (Rydosz et al., Materials 2019). "
        "Tout le reste (forme de Δφ(C), interféromètre) est **simulé**, et non mesuré.\n\n"
        "**À ne pas affirmer** : que l'interféromètre à CuO a été construit ; que V mesure C ; que 0–200 ppm est linéaire ; "
        "qu'on mesure un niveau global de pollution."
    )

# ---------------- G : vue 3D ----------------
def _batons(z, v, champ):
    """Segments verticaux entre l'axe de propagation et la courbe (effet « peigne » du champ)."""
    xs, ys, zs = [], [], []
    for zi, vi in zip(z, v):
        xs += [0, vi if champ == "E" else 0, None]
        ys += [0, vi if champ == "H" else 0, None]
        zs += [zi, zi, None]
    return xs, ys, zs


def _anim(fig_data, frames, titre, aspect):
    bouton = dict(type="buttons", x=0.0, y=0.0, buttons=[
        dict(label="▶ Lecture", method="animate",
             args=[None, {"frame": {"duration": 70, "redraw": True}, "fromcurrent": True, "transition": {"duration": 0}}]),
        dict(label="⏸ Pause", method="animate",
             args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]),
    ])
    fig = go.Figure(data=fig_data, frames=frames)
    fig.update_layout(
        title=titre, height=620, margin=dict(l=0, r=0, t=40, b=0), updatemenus=[bouton],
        scene=dict(xaxis=dict(title="E (u.a.)", range=[-2.6, 2.6]), yaxis=dict(title="H / voies", range=[-3.2, 3.2]),
                   zaxis=dict(title="z (m) : sens de propagation"), aspectmode="manual", aspectratio=aspect),
        legend=dict(orientation="h", y=1.02),
    )
    return fig


with tabs[6]:
    st.subheader("Visualisation 3D (tourne la figure avec la souris, ▶ pour animer)")
    vue = st.radio("Vue", ["Onde électromagnétique (E et H)", "Interféromètre : voies et somme"], horizontal=True, key="vue3d")
    g1, g2, g3 = st.columns(3)
    a3 = g1.slider("α (Np/m)", 0.0, 6.0, 0.8, 0.1, key="a3")
    b3 = g2.slider("β (rad/m)", 4.0, 40.0, 15.0, 1.0, key="b3")
    L3 = g3.slider("Longueur L (m)", 1.0, 5.0, 3.0, 0.5, key="L3")
    z = np.linspace(0, L3, 400)
    zb = np.linspace(0, L3, 60)
    wts = np.deg2rad(np.arange(0, 360, 10))

    if vue.startswith("Onde"):
        def dyn(wt):
            E = ph.onde(z, wt, a3, b3); Eb = ph.onde(zb, wt, a3, b3)
            ex, ey, ez = _batons(zb, Eb, "E"); hx, hy, hz = _batons(zb, Eb, "H")
            return [
                go.Scatter3d(x=E, y=0 * z, z=z, mode="lines", line=dict(color="crimson", width=6), name="E (champ électrique, plan x–z)"),
                go.Scatter3d(x=0 * z, y=E, z=z, mode="lines", line=dict(color="royalblue", width=6), name="H (champ magnétique, plan y–z)"),
                go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", line=dict(color="crimson", width=2), showlegend=False),
                go.Scatter3d(x=hx, y=hy, z=hz, mode="lines", line=dict(color="royalblue", width=2), showlegend=False),
            ]
        env = np.exp(-a3 * z)
        stat = [
            go.Scatter3d(x=0 * z, y=0 * z, z=z, mode="lines", line=dict(color="black", width=3), name="axe de propagation z"),
            go.Scatter3d(x=env, y=0 * z, z=z, mode="lines", line=dict(color="gray", dash="dash", width=3), name="enveloppe e^(−αz)"),
            go.Scatter3d(x=-env, y=0 * z, z=z, mode="lines", line=dict(color="gray", dash="dash", width=3), showlegend=False),
        ]
        frames = [go.Frame(data=dyn(w), traces=[0, 1, 2, 3], name=str(i)) for i, w in enumerate(wts)]
        fig = _anim(dyn(0) + stat, frames, "Onde progressive avec pertes", dict(x=1, y=1, z=2.4))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("E oscille dans un plan, H dans le plan perpendiculaire, et l'onde avance selon z en s'amortissant (enveloppe). "
                   "H est tracé en phase avec E et à la même échelle (milieu supposé peu dispersif) : c'est une représentation pédagogique.")
    else:
        dphi_deg = st.slider("Déphasage Δφ entre les deux voies (°)", 0, 360, int(np.rad2deg(biais)), 1, key="dphi3d")
        dph = np.deg2rad(dphi_deg)

        def dyn2(wt):
            r = ar * np.cos(wt - b3 * z); m = am * np.cos(wt - b3 * z + dph)
            s = 0.5 * (r + m)
            return [
                go.Scatter3d(x=r, y=0 * z - 2.2, z=z, mode="lines", line=dict(color="seagreen", width=6), name="voie de référence"),
                go.Scatter3d(x=m, y=0 * z, z=z, mode="lines", line=dict(color="darkorange", width=6), name="voie de mesure (avec capteur CuO)"),
                go.Scatter3d(x=s, y=0 * z + 2.2, z=z, mode="lines", line=dict(color="black", width=6), name="sortie = (réf + mesure)/2"),
            ]
        stat = [go.Scatter3d(x=0 * z, y=0 * z + y0, z=z, mode="lines", line=dict(color="lightgray", width=2), showlegend=False)
                for y0 in (-2.2, 0, 2.2)]
        frames = [go.Frame(data=dyn2(w), traces=[0, 1, 2], name=str(i)) for i, w in enumerate(wts)]
        fig = _anim(dyn2(0) + stat, frames, "Deux voies cohérentes recombinées", dict(x=1, y=1.6, z=2.4))
        st.plotly_chart(fig, use_container_width=True)
        P = float(ph.puissance(ar, am, dph))
        st.write(f"Δφ = **{dphi_deg}°** → Pₛ/P₀ = **{P:.4f}** · contraste V = **{ph.contraste(ar, am):.4f}**")
        st.caption("À Δφ = 0° les deux ondes s'additionnent (sortie maximale) ; à 180° elles s'annulent ; à 90° on est au point le plus sensible. "
                   "Dans le vrai montage, c'est l'acétone qui fait varier Δφ de quelques degrés seulement.")
