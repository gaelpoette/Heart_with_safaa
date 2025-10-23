# coding : utf-8
import numpy as np
import matplotlib.pyplot as plt
import json
from scipy.optimize import curve_fit
from matplotlib.backends.backend_pdf import PdfPages

plt.close('all')
plt.ion()

# Some durations and voltages, depending on the stimulator used
BOREA_D = np.array([
    0.12, 0.25, 0.35, 0.50, 0.60, 0.75, 0.85, 1.00
])

BOREA_V = np.array([
    0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.0
])

PSA_D = np.array([
    0.06, 0.12, 0.18, 0.24, 0.3, 0.36, 0.42, 0.48, 0.54, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 1.8, 1.9, 2.0
])

PSA_V = np.array([
    0.0,  0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10, 10
])


def f(duration, rheo, chro):
    """Lapicque function"""
    return chro*rheo/duration+rheo


def fit_data(durations, uppers, lowers):
    """Curve fit"""
    mids = [0.5*(uppers[i]+lowers[i]) for i in range(len(uppers))]
    popt,pcov = curve_fit(f, durations, mids, bounds=(0, np.inf))
    fff,aaa = plt.subplots()
    aaa.scatter(durations,mids)
    ddd = np.linspace(0.05,10,100)
    aaa.plot(ddd,f(ddd,*popt))
    perr = np.sqrt(np.diag(pcov))
    print("Ch",popt[1],"errC",perr[1]/popt[1],"Rh",popt[0],"errR",perr[0]/popt[1])
    return popt[1],perr[1],popt[0],perr[0]


# Read the data
exp_data = json.load(open("./data_experiments.json",'r'))
sim_data = json.load(open("./data_simulations.json",'r'))
# SIM/EXP with Borea voltages 
# ceps_data    = sim_data["ceps_results" ][1]
# circe_data   = sim_data["circe_results"][4]
# exps_to_plot = [("sheep2",0)]
# CEPS_V       = np.array([0.0,0.25,0.5,0.75,1.00,1.25,1.50,1.75,2.00,2.50,3.00])
# CIRCE_V      = np.array([0.0,0.25,0.5,0.75,1.0,1.25,1.50,1.75,2.00,2.50,3.00,3.50,4.00])
# outputPng    = ("../img/compLapicqueBorea.png")
# SIM/EXP with PSA voltages
ceps_data    = sim_data["ceps_results" ][0]
circe_data   = sim_data["circe_results"][6]
exps_to_plot = [("sheep3",0),("sheep3",3),("sheep4",0),("sheep4",2)]
CEPS_V       = PSA_V[PSA_V<3.1]
CIRCE_V      = PSA_V[PSA_V<3.1]
outputPng    = ("../img/compLapicquePSA.png")

# All PSA exp
exps_to_plot = [("sheep3",0),
                ("sheep3",1),
                ("sheep3",3),
                ("sheep4",0),
                ("sheep4",1),
                ("sheep4",2),
                ("sheep4",3),
                ("sheep5",0),
                ("sheep5",1),
                ("sheep5",2),
                ("sheep5",3),
                ("sheep6",0),
                ("sheep6",1),
                ("sheep7",0),
                ("sheep7",1),
                ("sheep7",2),
                ("sheep7",3)
]
# exps_to_plot = [("sheep3",0),
#                 ("sheep3",3),
#                 ("sheep4",0),
#                 ("sheep4",2)
# ]


# For braces
bl1 = 0.02
bl2 = bl1/5.

c0D = '#CC4B00'
c3D = '#550C11'
colors = ['#0C285F','brown',"green",'indigo','orangered']

fig, ax = plt.subplots(figsize=(4,3))
chros = []
rheos = []
perrc = []
perrr = []
for iexp,exp_to_plot in enumerate(exps_to_plot):

  name,search_index = exp_to_plot

  exp_to_plot = exp_data[name]["searches"][search_index]

  VOLT = PSA_V if "PSA" in exp_to_plot["comments"] else BOREA_V
  DURS = PSA_D if "PSA" in exp_to_plot["comments"] else BOREA_D
  dmin, dmax = np.min(DURS), np.max(DURS)

  durs   = exp_to_plot["durations"]
  upper  = exp_to_plot["thresholds"]
  lower  = [VOLT[np.argmin(np.abs(VOLT-v))-1] for v in upper]

  ax.fill_between(durs, lower, upper, color=colors[iexp%len(colors)],
                  alpha=0.25, label=f"{name}, search #{search_index+1}")

  chro,errc,rheo,errr = fit_data(durs,upper,lower)
  chros.append(chro)
  perrc.append(errc)
  rheos.append(rheo)
  perrr.append(errr)


durs      = ceps_data["durations" ]
upper     = ceps_data["thresholds"]
lower     = [CEPS_V[np.argmin(np.abs(CEPS_V-v))-1] for v in upper]
for i in range(len(upper)):
  if i==0:
    ax.plot([durs[i],durs[i]],[lower[i],upper[i]],color=c3D,label="3D model")
  else:
    ax.plot([durs[i],durs[i]],[lower[i],upper[i]],color=c3D)
  ax.plot([durs[i]-bl1/2.,durs[i]-bl1/2.,durs[i]+bl1/2.,durs[i]+bl1/2.],
          [upper[i]-bl2,upper[i],upper[i],upper[i]-bl2],color=c3D)
  ax.plot([durs[i]-bl1/2.,durs[i]-bl1/2.,durs[i]+bl1/2.,durs[i]+bl1/2.],
          [lower[i]+bl2,lower[i],lower[i],lower[i]+bl2],color=c3D)
  

chros = np.asarray(chros)
perrc = np.asarray(perrc)
rheos = np.asarray(rheos)
perrr = np.asarray(perrr)

print("Chros, errc, rheos, err r")
print("avg",np.mean(chros),np.mean(perrc),np.mean(rheos),np.mean(perrr))
print("std",np.std (chros),np.std (perrc),np.std (rheos),np.std (perrr))
print("min",np.min (chros),np.min (perrc),np.min (rheos),np.min (perrr))
print("max",np.max (chros),np.max (perrc),np.max (rheos),np.max (perrr))

print("CEPS",)
fit_data(durs,upper,lower)

durs      = circe_data["durations" ]
upper     = circe_data["thresholds"]
lower     = [CIRCE_V[np.argmin(np.abs(CIRCE_V-v))-1] for v in upper]

for i in range(len(upper)):
  if i==0:
    ax.plot([durs[i],durs[i]],[lower[i],upper[i]],color=c0D,label="0D model")
  else:
    ax.plot([durs[i],durs[i]],[lower[i],upper[i]],color=c0D)
  ax.plot([durs[i]-bl1/2.,durs[i]-bl1/2.,durs[i]+bl1/2.,durs[i]+bl1/2.],
          [upper[i]-bl2,upper[i],upper[i],upper[i]-bl2],color=c0D)
  ax.plot([durs[i]-bl1/2.,durs[i]-bl1/2.,durs[i]+bl1/2.,durs[i]+bl1/2.],
          [lower[i]+bl2,lower[i],lower[i],lower[i]+bl2],color=c0D)
print("Circe ",)
fit_data(durs,upper,lower)

ax.set_xlabel("Duration (ms)")
ax.set_ylabel("Voltage (V)")

ax.legend()
ax.grid(color="0.85")

# fig.tight_layout()
# fig.savefig(outputPng,dpi=300)




# Loop through the data
# all_curves = {}

# for sheep_index, (sheep_name, sheep_data) in enumerate(exp_data.items()):
#     sheep_date = sheep_data.get("date")
#     sheep_comments = sheep_data.get("comments")
#     searches = sheep_data.get("searches")
#     n_searches = len(searches)

#     pp = PdfPages(f'sheep_{sheep_index+1}.pdf')

#     print("> sheep {} (date {}), n searches = {}".format(
#         sheep_index+1, sheep_date, n_searches))

#     for search_index, search_data in enumerate(searches):
#         comments = search_data.get("comments")
#         order = search_data.get("order")
#         n_spikes = search_data.get("n spikes")
#         bpm = search_data.get("bpm")
#         site = search_data.get("site")
#         tissue = search_data.get("tissue")
#         bath = search_data.get("bath")
#         impedance = search_data.get("impedance")
#         durs = np.array(search_data.get("durations"))
#         thrs = np.array(search_data.get("thresholds"))

#         n_durations = np.size(durs)
#         assert n_durations == np.size(
#             thrs), f"incompatible sizes (site {site}, order {order})"

#         print("\t> site {}, tissue {}, bpm {}, n {}".format(
#             site, tissue, bpm, n_durations))

#         VOLT = PSA_V if "PSA" in comments else BOREA_V
#         DURS = PSA_D if "PSA" in comments else BOREA_D
#         dmin, dmax = np.min(DURS), np.max(DURS)

#         indices = np.argmin(np.abs(thrs[:, None]-VOLT), axis=1)

#         if "A" in order:
#             # in ascending mode it’s the first capturing voltage
#             idx = np.maximum(indices-1, 0)
#             upper = VOLT[idx]
#             below = np.copy(thrs)
#         else:
#             # in descending mode it’s the last capturing voltage
#             idx = np.maximum(indices-1, 0)
#             upper = np.copy(thrs)
#             below = VOLT[idx]

#         middle = (below + upper) / 2.0
#         delta = upper - middle
#         x_optimal, y_optimal, a, b = fit_data(dmin, dmax, durs, middle, delta)

#         x_e = durs
#         y_e = middle
#         delta = y_e - np.interp(x_e, x_optimal, y_optimal)
#         mask = np.sign(delta)
#         uplim = mask > 0.0
#         lolim = mask < 0.0

#         delta = np.abs(delta)

#         if "PSA" in comments:
#             if "healthy" in tissue:
#                 all_curves[f"Sheep {sheep_index+1}, {site}, {tissue}, {order}, {comments}"] = {
#                     "x": x_optimal, "y": y_optimal,
#                     "x_e": x_e, "y_e": y_e, "delta": delta, "uplim": uplim, "lolim": lolim,
#                     "a": a, "b": b
#                 }

#         fig, ax = plt.subplots()
#         ax.fill_between(durs, below, upper, color='b',
#                         alpha=0.25, label="regions")
#         ax.plot(durs, middle, '--b', label="experimental")
#         ax.plot(x_optimal, y_optimal, 'r',
#                 label="a/x+b ; a={:.2f}, b={:.2f}".format(a, b))
#         ax.set_title("Sheep {}, site {}, tissue {}, bpm {}".format(
#             sheep_index+1, site, tissue, bpm))
#         ax.set_xlim(np.min(DURS)-0.1, np.max(DURS)+0.1)
#         ax.set_ylim(np.min(VOLT)-0.1, np.max(VOLT)+0.1)
#         ax.legend()
#         pp.savefig(fig)
#         plt.close(fig)

#     # plt.show()
#     pp.close()

# fig, ax = plt.subplots()
# for name, search in all_curves.items():
#     # ax.plot(search["x"], search["y"], label=name)
#     plt.errorbar(search["x_e"], search["y_e"],
#                  yerr=search["delta"], uplims=search["uplim"], lolims=search["lolim"], label=name)
# ax.legend()

# fig, ax = plt.subplots()
# for name, search in all_curves.items():
#     ax.plot(search["x"], search["y"], label=name)
# ax.legend()
# plt.show()
