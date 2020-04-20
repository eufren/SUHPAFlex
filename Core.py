import openmdao.api as om
from Aerodynamics import Aero
from Structures import Struct
from Cables import CableDrag, CableSize
import scipy.interpolate as scipl


class LiftDeflectionCorrection(om.ExplicitComponent):

    def setup(self):

        self.add_discrete_input('lFunc', val=lambda x: x)
        self.add_discrete_input('dnudxFunc', val=lambda x: x)

        self.add_output('corrected_L')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        pass

        # TODO: Implement correction of lift based on deflection


class DragFunction(om.ExplicitComponent):

    def setup(self):

        self.add_input('D_wing')
        self.add_input('D_cable')

        self.add_output('D_total')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        D_wing = inputs['D_wing']
        D_cable = inputs['D_cable']

        D_total = D_wing + D_cable
        outputs['D_total'] = D_total


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

        # Attach analysis components to group.
        aeroSystem = self.add_subsystem('aeroSystem', Aero(),
                                        promotes_inputs=['b1', 'b2', 'b3', 'cr', 'ct'],
                                        promotes_outputs=['lFunc', 'S', 'L', 'D_wing'])

        # Define the cycle between the cable sizer and the structural analysis (linked by cable radius, cable force)
        structuralCycle = self.add_subsystem('structCycle', om.Group(), promotes=['*'])
        structuralCycle.add_subsystem('structSystem', Struct(),
                                      promotes_inputs= ['lFunc', 'cablePosition', 'cableRadius'],
                                      promotes_outputs= ['nuFunc', 'dnudxFunc', 'cableForce'])
        structuralCycle.add_subsystem('cableSizer', CableSize(),
                                      promotes_inputs= ['cableForce', 'cablePosition'],
                                      promotes_outputs= ['cableRadius'])
        structuralCycle.nonlinear_solver = om.NonlinearBlockGS()  # Configure an appropriate solver for the cycle.
        structuralCycle.set_order(['structSystem', 'cableSizer'])

        # Attach the rest of the components.
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
        self.set_order(['designVars', 'aeroSystem', 'structCycle', 'cableDrag', 'liftCorrection', 'totalDrag'])


        # Add constraint functions
        self.add_subsystem('spanConstraint', om.ExecComp('span = 2*( b1 + b2 + b3 )'), promotes=['*'])
        self.add_subsystem('rootThicknessConstraint', om.ExecComp('rootThickness = 0.14 * cr'), promotes=['*'])
        self.add_subsystem('tipThicknessConstraint', om.ExecComp('tipThickness = 0.11 * ct'), promotes=['*'])

# Create optimisation problem
prob = om.Problem()
prob.model = SUHPAFlex()

# Configure problem driver
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
prob.driver.options['maxiter'] = 100
prob.driver.options['tol'] = 1e-8

# Add design variables, constraints and objective
prob.model.add_design_var('b1', lower=0.5, upper=12)
prob.model.add_design_var('b2', lower=0.5, upper=12)
prob.model.add_design_var('b3', lower=0.5, upper=12)
prob.model.add_design_var('cr', lower=0.2, upper=2)
prob.model.add_design_var('ct', lower=0.2, upper=2)
prob.model.add_design_var('cablePosition', lower=0.1, upper=11.9)

prob.model.add_constraint('corrected_L', equals=1.231)
prob.model.add_constraint('span', equals=24)
prob.model.add_constraint('rootThickness', lower=0.1)
prob.model.add_constraint('tipThickness', lower=0.03)

prob.model.add_objective('D_total')

# Start the optimiser
prob.setup()

#prob.check_config()

prob.run_model()
# prob.run_driver()
#
# print("Optimised design variables:")
# print(prob['b1'])
# print(prob['b2'])
# print(prob['b3'])
# print(prob['cr'])
# print(prob['ct'])
# print(prob['cablePosition'])
#
# print("Optimised lift, wing drag, cable drag and total drag:")
# print(prob['corrected_L'])
# print(prob['D_wing'])
# print(prob['D_cable'])
# print(prob['D_total'])

