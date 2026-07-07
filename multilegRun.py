import openmm as mm
from openmm import app
from openmm.app import *
from openmm.unit import *
from sys import argv

num_repeats = int(argv[1])

pdb = PDBFile('input.pdb')
forcefield = ForceField('amber99sb.xml', 'tip3p.xml')
system = forcefield.createSystem(pdb.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

for i in range(num_repeats):
    simulation = Simulation(pdb.topology, system, integrator)
    if i == 0:
        simulation.context.setPositions(pdb.positions)
    else:
        with open(f'final_state_{i-1}.xml', 'r') as f:
            state = mm.XmlSerializer.deserialize(f.read())
        simulation.context.setState(state)
    simulation.reporters.append(DCDReporter(f'traj_{i}.dcd', 1000))
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True, potentialEnergy=True, temperature=True))
    simulation.step(25000000)
    simulation.saveState(f'final_state_{i}.xml')