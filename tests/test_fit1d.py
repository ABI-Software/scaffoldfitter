from cmlibs.maths.vectorops import add
from cmlibs.utils.zinc.field import find_or_create_field_coordinates, find_or_create_field_group
from cmlibs.utils.zinc.finiteelement import create_hermite_curve_elements, create_node_points
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.context import Context
from cmlibs.zinc.field import Field
from cmlibs.zinc.node import Node
from scaffoldfitter.fitter import Fitter
from scaffoldfitter.fitterstepfit import FitterStepFit

import logging
import os
import sys
import unittest


here = os.path.abspath(os.path.dirname(__file__))


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class GeneralTestCase(unittest.TestCase):

    def test_fit_1d_vagus_branch_stretch_instability(self):
        """
        Test 1D fit of vagus nerve branch which is sensitive to small perturbations to
        projections at the ends of the branch which converge to quite different
        results based due to stretch being applied by a knife-edge condition if
        end point projections are in or out of the element.
        """
        # sensitive data for branch: left A thoracic cardiopulmonary branch of vagus nerve
        # from scaffoldmaker nerve1 test
        ax = [
            [20614.63786414724, -2939.873650361369, -605.4150310438267],
            [19866.346579952475, -4452.963739972778, -662.9986048963924],
            [18688.55297975595, -5664.535791350198, -711.1634865902475]]
        ad1 = [
            [-490.70985052735557, -1622.8873929937577, -60.65382802572525],
            [-984.464558558513, -1380.6496536715067, -53.615308968309854],
            [-1342.5638510703225, -1036.3223530813373, -42.38892426110936]]
        px = [
            [20614.63786414724, -2939.873650361369, -605.4150310438267],
            [20651.94765027246, -2955.453126748926, -609.7187313635989],
            [20647.32974677242, -3069.288380278588, -613.2407621141546],
            [20633.4170487354, -3181.933865240825, -616.8148329783874],
            [20610.80328363998, -3293.108436310056, -620.4267075925841],
            [20580.08217896472, -3402.530948160697, -624.0621495930316],
            [20541.84746218819, -3509.920255467168, -627.7069226160168],
            [20496.69286078899, -3614.995212903887, -631.3467902978265],
            [20445.21210224565, -3717.474675145269, -634.9675162747475],
            [20387.99891403679, -3817.077496865736, -638.5548641830671],
            [20325.64702364096, -3913.522532739701, -642.094597659072],
            [20258.75015853673, -4006.528637441586, -645.5724803390489],
            [20188.24413551214, -4096.420788564372, -648.9895420778794],
            [20114.90411379834, -4183.900815486511, -652.3598548674369],
            [20039.08283453852, -4269.252850560641, -655.6887455494957],
            [19961.13303887584, -4352.761026139395, -658.9815409658308],
            [19881.4074679535, -4434.709474575406, -662.2435679582161],
            [19800.25886291468, -4515.382328221309, -665.4801533684263],
            [19718.03996490257, -4595.063719429735, -668.6966240382361],
            [19635.10351506035, -4674.037780553322, -671.8983068094196],
            [19551.80225453122, -4752.588643944701, -675.0905285237516],
            [19468.48892445835, -4831.000441956509, -678.2786160230067],
            [19385.29599178565, -4909.541186249976, -681.4694983759377],
            [19302.34830367641, -4988.461861449508, -684.6696568713676],
            [19219.98717160285, -5068.021059352487, -687.8837466811756],
            [19138.55390703724, -5148.477371756298, -691.1164229772405],
            [19058.38982145181, -5230.089390458326, -694.3723409314412],
            [18979.83622631881, -5313.11570725596, -697.6561557156568],
            [18903.23443311048, -5397.814913946578, -700.9725225017663],
            [18828.92575329906, -5484.445602327566, -704.3260964616486],
            [18757.2514983568, -5573.266364196313, -707.7215327671828],
            [18688.55297975595, -5664.535791350198, -711.1634865902475]]

        mag = 1.0E-12
        offsets = [
            [0.0, 0.0, 0.0],
            [mag, 0.0, 0.0],
            [-mag, 0.0, 0.0],
            [0.0, mag, 0.0],
            [0.0, -mag, 0.0],
            [0.0, 0.0, mag],
            [0.0, 0.0, -mag]
        ]

        for offset in offsets:
            context = Context("Test branch fit instability")
            region = context.getDefaultRegion()

            fieldmodule = region.getFieldmodule()
            with ChangeManager(fieldmodule):
                coordinates = find_or_create_field_coordinates(fieldmodule)
                group = find_or_create_field_group(fieldmodule, "branch")
                mesh1d = fieldmodule.findMeshByDimension(1)
                mesh_group = group.getOrCreateMeshGroup(mesh1d)
                create_hermite_curve_elements(mesh_group, ax, ad1)

                datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
                data_group = group.getOrCreateNodesetGroup(datapoints)
                tx = [add(offset, x) for x in px]
                create_node_points(data_group, tx)

                # need a zero fibre field to apply strain/curvature penalties on 1-D model
                zero_fibres = find_or_create_field_zero_fibres(fieldmodule)

            fitter = Fitter(region=region)
            fitter.setDiagnosticLevel(0)
            fitter.setModelCoordinatesField(coordinates)
            fitter.setFibreField(zero_fibres)
            fitter.defineCommonMeshFields()
            fitter.setDataCoordinatesField(coordinates)
            fitter.defineDataProjectionFields()
            fitter.initializeFit()

            # calibrated for 25 points / element
            # data_weight = (25.0 * elements_count) / points_count
            # strain_penalty = 1.0E-6 * curve_length
            # # calibrated by scaling the model: a power of 3 relationship
            # curvature_penalty = 1.0E-6 * (curve_length ** 3)
            # values taken from scaffoldmaker run:
            data_weight = 1.5625
            strain_penalty = 0.003392933773782769
            curvature_penalty = 39059.4522237989
            fit1 = FitterStepFit()
            fitter.addFitterStep(fit1)
            fit1.setGroupDataWeight(None, data_weight)
            fit1.setGroupStrainPenalty(None, [strain_penalty])
            fit1.setGroupCurvaturePenalty(None, [curvature_penalty])
            fit1.run()
            self.assertTrue(fit1.hasRun())

            # extract fitted curve parameters from nodes
            expected_x = [
                [20637.12318111511, -2947.0943923264203, -608.014316560503],
                [19862.83638434499, -4453.08578556958, -662.8684815807565],
                [18689.987728765256, -5662.919125077371, -711.1721042445245]]
            expected_d1 = [
                [106.2553618007571, -1832.4602883200805, -65.28328438119757],
                [-1191.090913491253, -1202.5860376030116, -49.71124284349609],
                [-1014.2038163846042, -1392.7527811776654, -53.45575090415969]]
            fieldcache = fieldmodule.createFieldcache()
            nodes = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
            for n in range(3):
                node = nodes.findNodeByIdentifier(n + 1)
                fieldcache.setNode(node)
                result, x = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_VALUE, 1, 3)
                result, d1 = coordinates.getNodeParameters(fieldcache, -1, Node.VALUE_LABEL_D_DS1, 1, 3)
                for c in range(3):
                    self.assertAlmostEqual(x[c], expected_x[n][c], delta=1.0E-6)
                    self.assertAlmostEqual(d1[c], expected_d1[n][c], delta=1.0E-8)


def find_or_create_field_zero_fibres(fieldmodule, name="zero fibres", managed=True):
    """
    Find or create a constant field representing 3 zero fibre angles.
    :param fieldmodule: Owning field module to find or create field in.
    :param name: Name of zero fibres field.
    :param managed: Whether
    :return: Zinc Field.
    """
    zero_fibres = fieldmodule.findFieldByName(name)
    if zero_fibres.isValid():
        return zero_fibres
    with ChangeManager(fieldmodule):
        zero_fibres = fieldmodule.createFieldConstant([0.0, 0.0, 0.0])
        zero_fibres.setName(name)
        zero_fibres.setManaged(managed)
    return zero_fibres


if __name__ == "__main__":
    unittest.main()
