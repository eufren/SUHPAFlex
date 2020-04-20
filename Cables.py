import openmdao.api as om
import scipy.interpolate as scinpl
import numpy as np

class CableSize(om.ExplicitComponent):

    def setup(self):

        self.add_input('cableForce')
        self.add_input('cablePosition')

        self.add_output('cableRadius')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cableForce = inputs['cableForce']
        yieldStress = 370E+6

        outputs['cableRadius'] = (cableForce/(0.85*yieldStress*np.pi))**0.5


class CableDrag(om.ExplicitComponent):

    def setup(self):

        self.add_input('cableRadius')
        self.add_input('cablePosition')

        self.add_output('D_cable')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cableRadius = inputs['cableRadius']
        cablePosition = inputs['cablePosition']

        # Create lookup table for drag coefficient on cylinder
        Res = [1.0014079,1.440703296,2.011246642,2.894101489,4.165244883,6.367060117,10.24978986,14.52060946,20.38809078,29.86223903,48.47676582,67.64550951,109.8235046,162.8105771,227.1343507,400.5392623,620.6477051,914.5137619,1347.276005,2009.197036,3816.949487,6345.479294,9442.691856,14448.0294,21222.31987,30603.39122,42691.17072,61015.32321,85641.00388,123149.9875, 156870.7812]
        Cds = [11.30472478,9.459302224,7.487839772,5.867002856,4.326453502,3.493572625,2.945398548,2.341195418,2.046229131,1.856310053,1.717782833,1.569687516,1.402759235,1.288691372,1.277434834,1.207084039,1.122663682,1.042076055,1.028084527,0.981949789,0.958149093,1.018275729,1.120995651,1.174756542,1.191116918,1.194236286,1.21320866,1.216569694,1.193183453,1.175481069,1.219844662]
        dragCoefficientFunc = scinpl.interp1d(Res, Cds)

        # Aircraft properties
        h = 1.5 # Distance from anchor point of flying wire to spar.

        cableLength = 2*(h**2 + cablePosition**2)**0.5
        Re = (8.25*2*cableRadius) / 1.4207E-5
        Cd = dragCoefficientFunc(Re)
        D_cable = 0.5*1.225*(8.25**2)*(2*cableRadius)*cableLength*Cd

        outputs['D_cable'] = D_cable