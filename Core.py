import openmdao.api as om
from Aerodynamics import WingAero, DragFunction
from Structures import Deflections
from Cables import CableForce, CableDrag, CableSize
import numpy as np
import scipy.interpolate as scipl


class LiftDeflectionCorrection(om.ExplicitComponent):

    def setup(self):

        self.add_discrete_input('lFunc', val=lambda x: x)
        self.add_discrete_input('dnudxFunc', val=lambda x: x)

        self.add_output('corrected_L')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        lFunc = discrete_inputs['lFunc']
        dnudxFunc = discrete_inputs['dnudxFunc']

        step = 10E-3
        semispan = 12
        x = np.arange(0, semispan, step) # Generate array of spanwise stations.
        L = lFunc(x)
        dnudx = dnudxFunc(x)
        inclinationAngles = np.arctan(dnudx)
        corrected_L = 2*np.trapz(L*np.cos(inclinationAngles), x)

        outputs['corrected_L'] = corrected_L


class StructuralCycle(om.Group):

    def setup(self):
        # Link the structural components together (deflection, cable stress and cable sizing)
        deflectionSystem = self.add_subsystem('deflectionSystem', Deflections(),
                                              promotes_inputs=['lFunc', 'cablePosition', 'cableForce', 'cableHeight'],
                                              promotes_outputs=['nuFunc', 'dnudxFunc', 'cableStrain'])
        cableForce = self.add_subsystem('cableForce', CableForce(),
                                        promotes_inputs=['cableRadius', 'cableStrain'],
                                        promotes_outputs=['cableForce', 'cableStress'])
        cableSizing = self.add_subsystem('cableSizing', CableSize(),
                                         promotes_inputs=['cableForce'],
                                         promotes_outputs=['cableRadius'])

        # Choose an appropriate solver to converge the systems.
        self.nonlinear_solver = om.NonlinearBlockGS()
        self.linear_solver = om.LinearBlockGS()


class SUHPAFlex(om.Group):

    def setup(self):
        # Declaring design variables
        designVars = self.add_subsystem('designVars', om.IndepVarComp(), promotes=['*'])
        designVars.add_output('b1', 6)  # Default values are current SUHPA design
        designVars.add_output('b2', 4)
        designVars.add_output('b3', 2)
        designVars.add_output('cr', 1)
        designVars.add_output('ct', 0.4)
        designVars.add_output('cablePosition', 6)  # cablePosition is the point on the spar where the flying wire connects

        # Declaring fixed aircraft parameters
        parameters = self.add_subsystem('parameters', om.IndepVarComp(), promotes=['*'])
        parameters.add_output('cableHeight', 1.5)
        parameters.add_output('angleOfAttack', 4)
        parameters.add_output('flightSpeed', 8.25)
        parameters.add_output('airKinematicViscosity', 1.4207E-5)
        parameters.add_output('airDensity', 1.225)

        # Attach analysis components to group.
        aeroSystem = self.add_subsystem('aeroSystem', WingAero(),
                                        promotes_inputs=['b1', 'b2', 'b3', 'cr', 'ct', 'flightSpeed', 'angleOfAttack',
                                                         'airKinematicViscosity', 'airDensity'],
                                        promotes_outputs=['lFunc', 'S', 'L', 'D_wing'])

        structuralCycle = self.add_subsystem('structCycle', StructuralCycle(),
                                             promotes_inputs=['lFunc'],
                                             promotes_outputs=['dnudxFunc'])

        cableDrag = self.add_subsystem('cableDrag', CableDrag(),
                                       promotes_inputs= ['cableRadius', 'cablePosition'],
                                       promotes_outputs= ['D_cable'])

        liftCorrection = self.add_subsystem('liftCorrection', LiftDeflectionCorrection(),
                                            promotes_inputs= ['lFunc', 'dnudxFunc'],
                                            promotes_outputs= ['corrected_L'])

        totalDrag = self.add_subsystem('totalDrag', DragFunction(),
                                       promotes_inputs= ['D_wing', 'D_cable'],
                                       promotes_outputs= ['D_total'])

        # Set component execution order
        self.set_order(['designVars', 'parameters', 'aeroSystem', 'structCycle', 'cableDrag', 'liftCorrection', 'totalDrag'])

        # Add constraint functions
        self.add_subsystem('spanConstraint', om.ExecComp('span = 2*( b1 + b2 + b3 )'), promotes=['*'])
        self.add_subsystem('rootThicknessConstraint', om.ExecComp('rootThickness = 0.14 * cr'), promotes=['*'])
        self.add_subsystem('tipThicknessConstraint', om.ExecComp('tipThickness = 0.11 * ct'), promotes=['*'])

# Create optimisation problem
prob = om.Problem()
prob.model = SUHPAFlex()

# Configure problem driver
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'COBYLA'
prob.driver.options['maxiter'] = 100
prob.driver.options['tol'] = 1e-8

# Add design variables, constraints and objective
prob.model.add_design_var('b1', lower=0.5, upper=12)
prob.model.add_design_var('b2', lower=0.5, upper=12)
prob.model.add_design_var('b3', lower=0.5, upper=12)
prob.model.add_design_var('cr', lower=0.2, upper=2)
prob.model.add_design_var('ct', lower=0.2, upper=2)
prob.model.add_design_var('cablePosition', lower=0.1, upper=11)

prob.model.add_constraint('corrected_L', lower=1174, upper=1200)
prob.model.add_constraint('span', lower=23.9, upper=24.1)
prob.model.add_constraint('rootThickness', lower=0.1)
prob.model.add_constraint('tipThickness', lower=0.03)

prob.model.add_objective('D_total')

# Start the optimiser
prob.setup()

# prob.check_config()

prob.run_driver()

print("Optimised design variables:")
print(prob['b1'])
print(prob['b2'])
print(prob['b3'])
print(prob['cr'])
print(prob['ct'])
print(prob['cablePosition'])

print("Optimised lift, wing drag, cable drag and total drag:")
print(prob['corrected_L'])
print(prob['D_wing'])
print(prob['D_cable'])
print(prob['D_total'])

