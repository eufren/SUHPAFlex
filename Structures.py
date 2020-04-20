import openmdao.api as om
import numpy as np
import scipy.interpolate as scipl
import math


class Deflections(om.ExplicitComponent):

    def setup(self):

        self.add_discrete_input('lFunc', val=lambda x: 10 * (1 - ((2 * x) / (1.1 * 12 * 2)) ** 2) ** 0.5)
        self.add_input('cableHeight')
        self.add_input('cablePosition')
        self.add_input('cableForce', val=1000)

        self.add_discrete_output('nuFunc', val=lambda x: x)
        self.add_discrete_output('dnudxFunc', val=lambda x: 0.001 * x)
        self.add_output('cableStrain')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):

        lFunc = discrete_inputs['lFunc']
        cablePosition = inputs['cablePosition'][0]
        cableForce = inputs['cableForce'][0]
        h = inputs['cableHeight'][0]

        # Spar material properties, taken from Forrester's code
        ETensile = 170.000E+9
        EComp = 159.590E+9
        density = 1514.3

        # Spar properties, taken from Forrester's code
        semispan = 12
        width = 50E-3  # (Metres, 50mm)
        rootHeight = 100E-3
        tipHeight = 30E-3
        thicknessLower = 2E-3
        thicknessUpper = thicknessLower * (EComp / ETensile)

        def sparHeight(x):
            return rootHeight - (((rootHeight - tipHeight) / (semispan)) * x)

        def flexuralRigidityFunc(x):  # Ported from Forrester's code
            yUpper = sparHeight(x) / 2 - thicknessUpper / 2
            yLower = -(sparHeight(x) / 2 - thicknessLower / 2)
            IxUpper = (1 / 12) * width * thicknessUpper ** 3
            IxLower = (1 / 12) * width * thicknessLower ** 3
            AUpper = width * thicknessUpper
            ALower = width * thicknessLower
            neutralAxis = (yUpper * AUpper + yLower * ALower) / (AUpper + ALower)
            IxUpper = IxUpper + AUpper * (yUpper - neutralAxis) ** 2
            IxLower = IxLower + ALower * (yLower - neutralAxis) ** 2
            overallEI = ETensile * (IxUpper + IxLower)
            return overallEI

        def weightFunc(x):  # Note - this is weight per metre, and is integrated to get the weight force.
            return (thicknessLower + thicknessUpper) * width * density * 9.81

        def find_nearest(array, value):
            idx = np.searchsorted(array, value, side="left")
            if idx > 0 and (idx == len(array) or math.fabs(value - array[idx - 1]) < math.fabs(value - array[idx])):
                return idx - 1
            else:
                return idx

        step = 10E-3
        x = np.arange(0, semispan, step)  # Generate array of spanwise stations.
        n = len(x)
        L = lFunc(x)  # Generate array of lift forces per metre at each station.
        W = np.vectorize(weightFunc)(x)  # Generate array of weight forces per metre at each station.
        F = L - W
        EI = flexuralRigidityFunc(x)

        cableStation = find_nearest(x,cablePosition)  # Figure out which station is closest to the cable point.   # First guess at cable force
        cableAngle = np.arctan((h) / cablePosition)

        M = np.zeros(n)
        dnu2d2x = np.zeros(n)
        dnudx = np.zeros(n)
        nu = np.zeros(n)

        for i in range(n):
            M[i] = np.trapz(x[i:] * F[i:], x[i:]) - cableForce * np.sin(cableAngle) * (cablePosition - x[i] if x[i] < cablePosition else 0)
            dnu2d2x[i] = M[i] / EI[i]
            dnudx[i] = np.trapz(dnu2d2x[:i], x[:i])
            nu[i] = np.trapz(dnudx[:i], x[:i])

        cableAngle = np.arctan((h + nu[cableStation]) / cablePosition)
        if nu[cableStation] < 0:
            cableStrain = 0
        else:
            cableStrain = ((((h + nu[cableStation]) ** 2 + cablePosition ** 2) ** 0.5) / (
                        (h ** 2 + cablePosition ** 2) ** 0.5)) - 1

        print("Guess at cable strain: ", cableStrain)
        discrete_outputs['nuFunc'] = lambda x2: scipl.interp1d(x, nu)(x2)  # We return these as functions so each module can choose
        discrete_outputs['dnudxFunc'] = lambda x2: scipl.interp1d(x, dnudx)(x2)  # where it would like to sample.
        outputs['cableStrain'] = cableStrain