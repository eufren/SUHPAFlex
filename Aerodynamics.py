import openmdao.api as om
import LovelacePM as llpm
import scipy.interpolate as scipl
import numpy as np
import multiprocessing


class WingAero(om.ExplicitComponent):

    def setup(self):
        self.add_input('b1')
        self.add_input('b2')
        self.add_input('b3')
        self.add_input('cr')
        self.add_input('ct')
        self.add_input('flightSpeed')
        self.add_input('angleOfAttack')
        self.add_input('airKinematicViscosity')
        self.add_input('airDensity')

        self.add_discrete_output('lFunc', val=lambda x: 10*(1-((2*x)/(1.1*12*2))**2)**0.5)
        self.add_output('S')
        self.add_output('L')
        self.add_output('D_wing')

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        if multiprocessing.current_process().name !='LPM_child':

            b1 = inputs['b1'][0]
            b2 = inputs['b2'][0]
            b3 = inputs['b3'][0]
            cr = inputs['cr'][0]
            ct = inputs['ct'][0]
            Uinf = inputs['flightSpeed'][0]
            nu = inputs['airKinematicViscosity'][0]
            rho = inputs['airDensity'][0]
            alpha = inputs['angleOfAttack'][0]

            fx76 = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\openMDAO\SUHPAFlex2\Foils\fx76mp140"
            dae31 = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\openMDAO\SUHPAFlex2\Foils\dae31"
            fx76polar = llpm.read_polar(polname='fx76mp140', poldir = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\openMDAO\SUHPAFlex2\Foils", ext_append=True, echo=False)
            dae31polar = llpm.read_polar(polname='dae31', poldir = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\openMDAO\SUHPAFlex2\Foils", ext_append=True, echo=False)

            sld = llpm.Solid()

            root_sect = llpm.wing_section(afl=fx76, c=cr, xdisc=20, sweep=0, correction=fx76polar, Re=Uinf * cr / nu)

            left_b1 = llpm.wing_section(afl=fx76, c=cr, xdisc=20, sweep=0, correction=fx76polar, Re=Uinf * cr / nu,
                                        CA_position=np.array([0, -b1, 0]))

            left_b2 = llpm.wing_section(afl=fx76, c=cr, xdisc=20, sweep=0, correction=fx76polar,
                                        Re=Uinf * (cr - ((cr - ct) / (b2 + b3)) * b2) / nu,
                                        CA_position=np.array([0, -(b1 + b2), 0]))

            left_b3 = llpm.wing_section(afl=dae31, c=cr, xdisc=20, sweep=0, correction=dae31polar, Re=Uinf * ct / nu,
                                        CA_position=np.array([0, -(b1 + b2 + b3), 0]),
                                        incidence=-3.0, closed=True)

            right_b1 = llpm.wing_section(afl=fx76, c=cr, xdisc=20, sweep=0, correction=fx76polar, Re=Uinf * cr / nu,
                                         CA_position=np.array([0, b1, 0]))

            right_b2 = llpm.wing_section(afl=fx76, c=cr, xdisc=20, sweep=0, correction=fx76polar,
                                         Re=Uinf * (cr + ((cr - ct) / (b2 + b3)) * b2) / nu,
                                         CA_position=np.array([0, b1 + b2, 0]))

            right_b3 = llpm.wing_section(afl=dae31, c=cr, xdisc=20, sweep=0, correction=dae31polar, Re=Uinf * ct / nu,
                                         CA_position=np.array([0, b1 + b2 + b3, 0]),
                                         incidence=-3.0, closed=True)

            left_wingquad1 = llpm.wing_quadrant(sld, sect1=left_b3, sect2=left_b2)
            left_wingquad2 = llpm.wing_quadrant(sld, sect1=left_b2, sect2=left_b1)
            left_wingquad3 = llpm.wing_quadrant(sld, sect1=left_b1, sect2=root_sect)

            right_wingquad1 = llpm.wing_quadrant(sld, sect1=root_sect, sect2=right_b1)
            right_wingquad2 = llpm.wing_quadrant(sld, sect1=right_b1, sect2=right_b2)
            right_wingquad3 = llpm.wing_quadrant(sld, sect1=right_b2, sect2=right_b3)

            wng = llpm.wing(sld, wingquads=[left_wingquad1, left_wingquad2, left_wingquad3,
                                            right_wingquad1, right_wingquad2, right_wingquad3])

            acft = llpm.aircraft(sld, elems=[wng], echo=False)
            acft.edit_parameters({'a': alpha, 'Uinf': Uinf}, echo=False)
            wng.patchcompose(ydisc=120)
            acft.addwake()

            acft.eulersolve(echo=False)
            acft.calcforces(echo=False)

            S = wng.calc_reference()[0]  # Wing area.
            q = 0.5 * rho * Uinf**2  # Dynamic pressure, 1/2 * rho * V^2

            correctedLDistro = q * wng.Cls_corrected * wng.cs  # Includes viscous corrections.
            correctedL = q * S * (acft.CL + acft.dCL)
            correctedD = q * S * (acft.CD + acft.dCD)

            discrete_outputs['lFunc'] = lambda x: scipl.interp1d(np.append(wng.ys, 12), np.append(correctedLDistro, 0))(x)
            outputs['S'] = S
            outputs['L'] = correctedL
            outputs['D_wing'] = correctedD


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
