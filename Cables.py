import openmdao.api as om
import scipy.interpolate as scinpl
import numpy as np


class CableForce(om.ExplicitComponent):

    def setup(self):

        self.add_input('cableRadius')
        self.add_input('cableStrain')

        self.add_output('cableForce')
        self.add_output('cableStress')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cableRadius = inputs['cableRadius'][0]
        cableStrain = inputs['cableStrain'][0]

        ECable = 180E+9

        cableStress = ECable*cableStrain
        cableForce = cableStress*np.pi*cableRadius**2

        outputs['cableForce'] = cableForce
        outputs['cableStress'] = cableStress


class CableSize(om.ExplicitComponent):

    def setup(self):

        self.add_input('cableForce')

        self.add_output('cableRadius', val=3E-3)

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cableForce = inputs['cableForce']
        yieldStress = 370E+6

        cableRadius = (cableForce/(0.85*yieldStress*np.pi))**0.5
        outputs['cableRadius'] = cableRadius


class CableDrag(om.ExplicitComponent):

    def setup(self):

        self.add_input('cableRadius', val=3E-3)
        self.add_input('cablePosition')

        self.add_output('D_cable')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        cableRadius = inputs['cableRadius']
        cablePosition = inputs['cablePosition']

        # Create lookup table for drag coefficient on cylinder
        Res = [0.007349705,0.010580701,0.01303319,0.016279717,0.023290215,0.031551131,0.042741709,0.065063552,0.088915603,0.112862041,0.162408304,0.226734784,0.310823829,0.45554292,0.643701409,0.943457277,1.166800205,1.706878507,2.386873123,3.373127361,4.825004693,6.573384566,9.070756041,12.46009532,16.32264941,23.62573747,33.77486281,48.57895377,69.8805523,104.8704859,136.0705625,220.9061945,404.824181,578.6430232,837.3228216,1168.302505,1630.211008,2274.349921,3560.246161,4848.265348,7866.996616,10972.83599,16262.00159,23383.76157,33628.40214,48357.35243,69536.65349,100000.8229,143803.6045,206824.832,298828.3363,344954.8209,409812.0528,479262.8512,643129.4342,855059.7653,1276390.94,1780441.501,2638688.778,3793631.186,5455938.482,7797881.553,10620115.13,]
        Cds = [488.2185647,340.1526247,288.9031397,243.5301791,181.4068819,137.2151559,104.1398329,79.9940596,60.66165532,49.44656444,39.59782955,31.78771115,26.01348148,19.81329463,16.07027806,12.03102477,10.006122,8.10880111,6.559004767,5.129402538,4.003179054,3.468167037,2.89818539,2.439943096,2.228033798,1.917165257,1.816729979,1.699701291,1.527063884,1.390027872,1.343615357,1.2567531,1.152618439,1.146323723,1.073939138,1.039505286,0.986495013,0.990831385,0.958351797,0.954842499,1.055594039,1.146541754,1.198436977,1.222311177,1.199080147,1.208349974,1.222313385,1.200960102,1.201208122,1.142376541,1.001249159,0.789142422,0.471808603,0.308839797,0.278637232,0.301600501,0.354348603,0.37509318,0.390488554,0.4210001,0.406034219,0.415302917,0.399570865]
        dragCoefficientFunc = scinpl.interp1d(Res, Cds)

        # Aircraft properties
        h = 1.5 # Distance from anchor point of flying wire to spar.

        cableLength = 2*(h**2 + cablePosition**2)**0.5
        Re = (8.25*2*cableRadius) / 1.4207E-5
        Cd = dragCoefficientFunc(Re)
        D_cable = 0.5*1.225*(8.25**2)*(2*cableRadius)*cableLength*Cd

        outputs['D_cable'] = D_cable