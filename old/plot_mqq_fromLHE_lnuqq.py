import pylhe
import numpy as np
import matplotlib.pyplot as plt

# Path to your LHE file
lhe_file = "unweighted_events.lhe_muons"


m_lnu = []
m_qq  = []
nu_map = {
    11: 12,
    13: 14,
    15: 16
}
for event in pylhe.read_lhe(lhe_file):
    # Select final-state quarks
    quarks = [
        p for p in event.particles
        if p.status == 1 and abs(p.id) in [1, 2, 3, 4, 5, 6]
    ]
    leptons = [
        p for p in event.particles
        if p.status == 1 and abs(p.id) in [11,13,15]
    ]
    neutrinos = [
        p for p in event.particles
        if p.status == 1 and abs(p.id) in [12,14,16]
    ]



    # Loop over quark pairs
    for i in range(len(quarks)):
        for j in range(i + 1, len(quarks)):
            p1 = quarks[i]
            p2 = quarks[j]

            # 4-vector sum
            E  = p1.e  + p2.e
            px = p1.px + p2.px
            py = p1.py + p2.py
            pz = p1.pz + p2.pz

            m2 = E**2 - (px**2 + py**2 + pz**2)
            if m2 > 0:
                m_qq.append(np.sqrt(m2))

    for lep in leptons:
        expected_nu = nu_map[abs(lep.id)]
        for nu in neutrinos:
            if abs(nu.id) == expected_nu:

                E  = lep.e  + nu.e
                px = lep.px + nu.px
                py = lep.py + nu.py
                pz = lep.pz + nu.pz

                m2 = E**2 - (px**2 + py**2 + pz**2)
                if m2 > 0:
                    m_lnu.append(np.sqrt(m2))
                    
# Plot

plt.figure()
plt.hist(m_qq, bins=100, histtype="step")
plt.xlabel(r"$m_{qq}$ [GeV]")
plt.ylabel("Events")
plt.title("m_qq")
plt.tight_layout()
plt.show()

plt.figure()
plt.hist(m_lnu, bins=100, histtype="step")
plt.xlabel(r"$m_{\ell\nu}$ [GeV]")
plt.ylabel("Events")
plt.title("m_lnu")
plt.tight_layout()
plt.show()
