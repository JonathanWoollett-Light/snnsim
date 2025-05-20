import snntorch as snn
import torch
import itertools

# Part 1
# ------------------------------------------------------------------------------

# def leaky_integrate_neuron(U, time_step=1e-3, I=0, R=5e7, C=1e-10):
#   tau = R*C
#   U = U + (time_step/tau)*(-U + I*R)
#   return U

# num_steps = 100
# U = 0.9
# U_trace = []  # keeps a record of U for plotting

# for step in range(num_steps):
#   U_trace.append(U)
#   U = leaky_integrate_neuron(U)  # solve next step of U

# for i,v in enumerate(U_trace):
#   print(f"{i}: {v:.8f}")

# Part 2
# ------------------------------------------------------------------------------

time_step = 1e-3
R = 5
C = 1e-3

# leaky integrate and fire neuron, tau=5e-3
lif1 = snn.Lapicque(R=R, C=C, time_step=time_step)

# # Initialize membrane, input, and output
# mem = torch.ones(1) * 0.9  # U=0.9 at t=0
# cur_in = torch.zeros(num_steps, 1)  # I=0 for all t
# spk_out = torch.zeros(1)  # initialize output spikes

# # A list to store a recording of membrane potential
# mem_rec = [mem]

# # pass updated value of mem and cur_in[step]=0 at every time step
# for step in range(num_steps):
#   spk_out, mem = lif1(cur_in[step], mem)

#   # Store recordings of membrane potential
#   mem_rec.append(mem)

# # convert the list of tensors into one tensor
# mem_rec = torch.stack(mem_rec)

# for i,v in enumerate(list(itertools.chain(*mem_rec.tolist()))):
#   print(f"{i}: {v:.8f}")

# Part 3
# ------------------------------------------------------------------------------

# # Initialize input current pulse
# cur_in = torch.cat((torch.zeros(10, 1), torch.ones(190, 1)*0.1), 0)  # input current turns on at t=10

# # Initialize membrane, output and recordings
# mem = torch.zeros(1)  # membrane potential of 0 at t=0
# spk_out = torch.zeros(1)  # neuron needs somewhere to sequentially dump its output spikes
# mem_rec = [mem]
# spiked = []
# num_steps = 200

# # pass updated value of mem and cur_in[step] at every time step
# for step in range(num_steps):
#   spk_out, mem = lif1(cur_in[step], mem)
#   spiked.append(spk_out.item())
#   mem_rec.append(mem)

# # crunch -list- of tensors into one tensor
# mem_rec = torch.stack(mem_rec)

# for i,v in enumerate(list(itertools.chain(*mem_rec.tolist()))):
#   print(f"{i}: {v:.8f}")
# print(f"{spiked}")

# print(f"The calculated value of input pulse [A] x resistance [Ω] is: {cur_in[11]*lif1.R} V")
# print(f"The simulated value of steady-state membrane potential is: {mem_rec[200][0]} V")

# Part 4
# ------------------------------------------------------------------------------

num_steps = 200

# # Initialize current pulse, membrane and outputs
# cur_in1 = torch.cat((torch.zeros(10, 1), torch.ones(20, 1)*(0.1), torch.zeros(170, 1)), 0)  # input turns on at t=10, off at t=30
# mem = torch.zeros(1)
# spk_out = torch.zeros(1)
# mem_rec1 = [mem]


# # neuron simulation
# for step in range(num_steps):
#   spk_out, mem = lif1(cur_in1[step], mem)
#   mem_rec1.append(mem)
# mem_rec1 = torch.stack(mem_rec1)

# for i,v in enumerate(list(itertools.chain(*mem_rec1.tolist()))):
#   print(f"{i}: {v:.8f}")

# # Increase amplitude of current pulse; quarter the time.
# cur_in3 = torch.cat((torch.zeros(10, 1), torch.ones(5, 1)*0.147, torch.zeros(185, 1)), 0)  # input turns on at t=10, off at t=15
# mem = torch.zeros(1)
# spk_out = torch.zeros(1)
# mem_rec3 = [mem]

# # neuron simulation
# for step in range(num_steps):
#   spk_out, mem = lif1(cur_in3[step], mem)
#   mem_rec3.append(mem)
# mem_rec3 = torch.stack(mem_rec3)

# for i,v in enumerate(list(itertools.chain(*mem_rec3.tolist()))):
#   print(f"{i}: {v:.8f}")

# Part 5
# ------------------------------------------------------------------------------

# LIF w/Reset mechanism
def leaky_integrate_and_fire(mem, cur=0, threshold=1, time_step=1e-3, R=5.1, C=5e-3):
  tau_mem = R*C
  spk = (mem > threshold)
  mem = mem + (time_step/tau_mem)*(-mem + cur*R) - spk*threshold  # every time spk=1, subtract the threhsold
  return mem, spk

# Small step current input
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)
mem = torch.zeros(1)
mem_rec = []
spk_rec = []

# neuron simulation
for step in range(num_steps):
  mem, spk = leaky_integrate_and_fire(mem, cur_in[step])
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)
for i,v in enumerate(list(itertools.chain(*mem_rec.tolist()))):
  print(f"{i}: {v:.8f}")
for i,v in enumerate(list(itertools.chain(*spk_rec.tolist()))):
  print(f"{i}: {v:.8f}")