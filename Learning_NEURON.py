import threading
from turtle import delay
from neuron import n, gui
from neuron.units import ms, mV, um
import matplotlib.pyplot as plt

n.load_file("stdrun.hoc")

class Cell:
    '''
    Base class for cells. Handels morphology and biophysics setup, as well as positioning and rotation.
    '''
    def __init__(self, id_, x=0, y=0, z=0, theta=0):
        self.id = id_
        self.setup_morphology()
        self.all = self.soma.wholetree()
        self.setup_biophysics()
        self.x = x
        self.y = y
        self.z = z
        self.rotate_z(theta)
        n.define_shape()

    def __repr__(self):
        return f"{self.name}(id:{self.id})"

    def set_position(self, x, y, z):
        for sec in self.all:
            for i in range(sec.n3d()):
                sec.pt3dchange(i, x - self.x + sec.x3d(i), y - self.y + sec.y3d(i), z - self.z + sec.z3d(i), sec.diam3d(i))
        self.x = x
        self.y = y
        self.z = z

    def rotate_z(self, theta):
        for sec in self.all:
            for i in range(sec.n3d()):
                x = sec.x3d(i)
                y = sec.y3d(i)
                x_new = x * n.cos(theta) - y * n.sin(theta)
                y_new = x * n.sin(theta) + y * n.cos(theta)
                sec.pt3dchange(i, x_new, y_new, sec.z3d(i), sec.diam3d(i))

class BallAndStick(Cell):
    '''
    Ball and stick neuron model with Hodgkin-Huxley soma and passive dendrite. Inherits from Cell base class.
    '''
    name = "BallAndStick"

    def setup_morphology(self):
        self.soma = n.Section(name='soma')
        self.dend = n.Section(name='dend')
        self.dend.connect(self.soma)
        self.soma.L = self.soma.diam = 12.6157 * um # microns
        self.dend.L = 200 * um
        self.dend.diam = 1 * um

    def setup_biophysics(self):
        for sec in self.all:
            sec.Ra = 100  # Axial resistance in Ohm * cm
            sec.cm = 1  # Membrane capacitance in micro Farads / cm^2
        self.soma.insert(n.hh)
        for seg in self.soma:
            seg.hh.gnabar = 0.12  # Sodium conductance in S/cm2
            seg.hh.gkbar = 0.036  # Potassium conductance in S/cm2
            seg.hh.gl = 0.0003  # Leak conductance in S/cm2
            seg.hh.el = -54.3  # Reversal potential in mV
        # Insert passive current in the dendrite
        self.dend.insert(n.pas)
        for seg in self.dend:
            seg.pas.g = 0.001  # Passive conductance in S/cm2
            seg.pas.e = -65  # Leak reversal potential mV

def create_n_BallAndStick(n_cells, r):
    cells = []
    for i in range(n_cells):
        theta = i* 2 * n.PI / n_cells
        cells.append(BallAndStick(i, x=n.cos(theta)*r, y=n.sin(theta)*r, z=0, theta=theta))
    return cells

my_cells = create_n_BallAndStick(5, 50)
n.topology()

#ps = n.PlotShape(True)
#ps.show(0)
#threading.Event().wait()

stim = n.NetStim() # NetStim is an external spike generator that is used in a NetCon to stimulate synapses
syn_ = n.ExpSyn(my_cells[0].dend(0.5)) # ExpSyn is a synapse model that implements an exponential decay conductance change, mimicing an AMPA synapse

stim.number = 1
stim.start = 9 * ms
ncstim = n.NetCon(stim, syn_)
ncstim.delay = 1 * ms
ncstim.weight[0] = 0.04  # uS
syn_.tau = 2 * ms
# reverseal potential for syn_ can be accessed with the .e member

recording_cell = my_cells[0]
v_vec = n.Vector().record(recording_cell.soma(0.5)._ref_v)
t_vec = n.Vector().record(n._ref_t)
d_vec = n.Vector().record(recording_cell.dend(0.5)._ref_v)
syn_i = n.Vector().record(syn_._ref_i)

syns = []
netcons = []
for source, target in zip(my_cells, my_cells[1:]+[my_cells[0]]):
    syn = n.ExpSyn(target.dend(0.5))
    netcon = n.NetCon(source.soma(0.5)._ref_v, syn, sec=source.soma)
    netcon.weight[0] = 0.05  # uS
    netcon.delay = 5
    syns.append(syn)
    netcons.append(netcon)

spike_times = [n.Vector() for nc in netcons]
for nc, spike_t_vec in zip(netcons, spike_times):
    nc.record(spike_t_vec)

n.finitialize(-65*mV)
n.continuerun(100*ms)

plt.figure()
for i, spike_t_vec in enumerate(spike_times):
    plt.vlines(spike_t_vec.as_numpy(), i + 0.5, i + 1.5)
plt.xlabel("time (ms)")
plt.ylabel("Cell index")
plt.show()
''' Plotting synaptic conductance
fig = plt.figure(figsize=(8, 4))
ax1 = fig.add_subplot(2, 1, 1)
soma_plot = ax1.plot(t_vec, v_vec, color="black", label="soma(0.5)")
dend_plot = ax1.plot(t_vec, d_vec, color="red", label="dend(0.5)")
rev_plot = ax1.plot(
    [t_vec[0], t_vec[-1]], [syn_.e, syn_.e], label="syn reversal", color="blue", linestyle=":"
)
ax1.legend()
ax1.set_ylabel("mV")
ax1.set_xticks([])  # Use ax2's tick labels

ax2 = fig.add_subplot(2, 1, 2)
syn_plot = ax2.plot(t_vec, syn_i, color="blue", label="synaptic current")
ax2.legend()
ax2.set_ylabel(n.units("ExpSyn.i"))
ax2.set_xlabel("time (ms)")
plt.show()
'''
