import openmdao.api as om
import LovelacePM as llpm
import scipy.interpolate as scipl
import numpy as np


class Aero(om.ExplicitComponent):

    def setup(self):
        self.add_input('b1')
        self.add_input('b2')
        self.add_input('b3')
        self.add_input('cr')
        self.add_input('ct')

        self.add_output('clFunc')
        self.add_output('lFunc')
        self.add_output('S')
        self.add_output('L')
        self.add_output('D_wing')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b1 = float(inputs['b1'])
        b2 = float(inputs['b2'])
        b3 = float(inputs['b3'])
        cr = float(inputs['cr'])
        ct = float(inputs['ct'])

        print(b1, b2, b3, cr, ct)

        fx76 = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\openMDAO\SUHPAFlex2\Foils\fx76mp140"
        dae31 = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\openMDAO\SUHPAFlex2\Foils\dae31"
        fx76polar = llpm.read_polar(polname='fx76mp140', poldir = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\openMDAO\SUHPAFlex2\Foils", ext_append=True, echo=False)
        dae31polar = llpm.read_polar(polname='dae31', poldir = r"C:\Users\euana\Documents\Important Documents\Optimisation Project\openMDAO\SUHPAFlex2\Foils", ext_append=True, echo=False)

        Uinf = 8.25
        nu = 1.4207E-5

        sld = llpm.Solid(full_parallel=False)

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
        acft.edit_parameters({'a': 4, 'Uinf': 8.25}, echo=False)
        wng.patchcompose(ydisc=120)
        acft.addwake()

        print("Before solve:")
        print(wng.__dir__())
        print("-------------")

        acft.eulersolve(echo=False)
        acft.calcforces(echo=False)

        print("After solve:")
        print(wng.__dir__())
        print("-------------")

        S = wng.calc_reference()[0]  # Wing area.
        q = 0.5 * 1.225 * 8.25**2  # Dynamic pressure, 1/2 * rho * V^2

        correctedLDistro = q * S * wng.Cls_corrected * wng.cs  # Includes viscous corrections.
        correctedL = q * S * (acft.CL + acft.dCL)
        correctedD = q * S * (acft.CD + acft.dCD)

        outputs['clFunc'] = scipl.interp1d(wng.ys, wng.Cls_corrected)
        outputs['lFunc'] = scipl.interp1d(wng.ys, correctedLDistro)
        outputs['S'] = S
        outputs['L'] = correctedL
        outputs['D_wing'] = correctedD
