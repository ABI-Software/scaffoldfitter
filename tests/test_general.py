import math
import os
import unittest
from cmlibs.utils.zinc.field import createFieldMeshIntegral
from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.utils.zinc.region import write_to_buffer, read_from_buffer
from cmlibs.zinc.context import Context
from cmlibs.zinc.field import Field
from cmlibs.zinc.result import RESULT_OK
from scaffoldfitter.fitter import Fitter
from scaffoldfitter.fitterstepalign import FitterStepAlign
from scaffoldfitter.fitterstepconfig import FitterStepConfig
from scaffoldfitter.fitterstepfit import FitterStepFit

here = os.path.abspath(os.path.dirname(__file__))


class GeneralTestCase(unittest.TestCase):

    def test_fit_1d_outliers(self):
        """
        Test 1D fit of nerve path with and without outliers (data too far away).
        """
        zinc_model_file = os.path.join(here, "resources", "nerve_trunk_model.exf")
        zinc_data_file = os.path.join(here, "resources", "nerve_path_data.exf")

        fitter = Fitter(zinc_model_file, zinc_data_file)
        fitter.setDiagnosticLevel(1)

        fit1 = FitterStepFit()
        fitter.addFitterStep(fit1)
        fit1.setGroupStrainPenalty(None, [0.0001])
        fit1.setGroupCurvaturePenalty(None, [0.001])

        config1 = FitterStepConfig()
        fitter.addFitterStep(config1)

        fit2 = FitterStepFit()
        fitter.addFitterStep(fit2)

        for case in range(3):
            fitter.load()
            fieldmodule = fitter.getFieldmodule()
            coordinates = fitter.getModelCoordinatesField()
            # set the in-built zero fibres field to use penalties
            zeroFibreField = fieldmodule.findFieldByName("zero fibres")
            fitter.setFibreField(zeroFibreField)

            if case == 0:
                # no outlier filtering
                expectedActiveDataSize = 27  # 25 data points + 2 marker point
                expectedLength = 3.0487700371049233
                expectedRmsError = 0.033684088293952655
                expectedMaxError = 0.1495389726841688
            else:
                if case == 1:
                    # absolute outlier length applied to default group
                    config1.setGroupOutlierLength(None, 0.1)
                elif case == 2:
                    # relative outlier length applied to "trunk" group
                    config1.clearGroupOutlierLength(None)
                    config1.setGroupOutlierLength("trunk", -0.1)
                expectedActiveDataSize = 26  # one outlier has been filtered
                expectedLength = 3.0331818804905284
                expectedRmsError = 0.009620758125514172
                expectedMaxError = 0.017088084995279192

            fitter.run()

            # check number of active data points and length of fitted model
            activeDataNodeset = fitter.getActiveDataNodesetGroup()
            self.assertEqual(activeDataNodeset.getSize(), expectedActiveDataSize)
            lengthField = createFieldMeshIntegral(coordinates, fitter.getMesh(1), number_of_points=4)
            self.assertTrue(lengthField.isValid())
            fieldcache = fieldmodule.createFieldcache()
            result, length = lengthField.evaluateReal(fieldcache, 1)
            self.assertEqual(result, RESULT_OK)
            TOL = 1.0E-8
            self.assertAlmostEqual(length, expectedLength, delta=TOL)
            rmsError, maxError = fitter.getDataRMSAndMaximumProjectionError()
            self.assertAlmostEqual(rmsError, expectedRmsError, delta=TOL)  # sqrt(0.12)
            self.assertAlmostEqual(maxError, expectedMaxError, delta=TOL)
            min_jac_el, min_jac_value = fitter.getLowestElementJacobian()
            self.assertEqual(1, min_jac_el)
            self.assertAlmostEqual(0.0, min_jac_value, delta=TOL)

    def test_setting_group_outlier_length(self):
        zinc_model_file = os.path.join(here, "resources", "nerve_trunk_model.exf")
        zinc_data_file = os.path.join(here, "resources", "nerve_path_data.exf")

        fitter = Fitter(zinc_model_file, zinc_data_file)

        config1 = FitterStepConfig()
        fitter.addFitterStep(config1)

        config1.setGroupOutlierLength(None, -3.0)
        self.assertEqual(-1.0, config1.getGroupOutlierLength(None)[0])

    def test_fit_projection_subgroup(self):
        # index i chooses one of 2 ways to run the test:
        # 0. From supplied model data files
        # 1. With a user-supplied region into which the user builds model, loads data and performs fit
        for i in range(2):
            zinc_model_file_name = os.path.join(here, "resources", "nerve_box.exf")
            zinc_data_file_name = os.path.join(here, "resources", "nerve_path_data.exf")
            if i == 0:
                # use fitter with model and data files
                region = None
                fitter = Fitter(zinc_model_file_name, zinc_data_file_name)
                fitter.setDiagnosticLevel(1)
                fitter.load()
                region = fitter.getRegion()
                fieldmodule = fitter.getFieldmodule()
                coordinates = fitter.getModelCoordinatesField()
            else:
                # use fitter with user-specified region; caller must build model, load data and set up fit
                context = Context("Scaffoldfitter test")
                region = context.getDefaultRegion()
                fitter = Fitter(region=region)
                fitter.setDiagnosticLevel(1)

                region.readFile(zinc_model_file_name)
                fieldmodule = region.getFieldmodule()
                coordinates = fieldmodule.findFieldByName("coordinates").castFiniteElement()
                self.assertEqual(coordinates.getNumberOfComponents(), 3)
                fitter.setModelCoordinatesField(coordinates)
                trunkGroup = fieldmodule.findFieldByName("trunk").castGroup()
                fitter.setModelFitGroup(trunkGroup)
                fitter.defineCommonMeshFields()

                dataRegion = region.createChild("raw_data")
                dataRegion.readFile(zinc_data_file_name)
                fitter.transferData(dataRegion)
                fitter.setDataCoordinatesField(coordinates)
                markerGroup = fieldmodule.findFieldByName("marker").castGroup()
                if markerGroup.isValid():
                    fitter.setMarkerGroup(markerGroup)
                fitter.defineDataProjectionFields()
                fitter.initializeFit()

            config0 = fitter.getInitialFitterStepConfig()
            self.assertEqual((None, False, False), config0.getGroupProjectionSubgroup(None))
            centroid = fieldmodule.findFieldByName("centroid").castGroup()
            self.assertTrue(centroid.isValid())
            config0.setGroupProjectionSubgroup(None, centroid)
            self.assertEqual((centroid, True, False), config0.getGroupProjectionSubgroup(None))
            # don't want centroid to apply to edge group
            self.assertEqual((centroid, False, True), config0.getGroupProjectionSubgroup("edge"))
            config0.setGroupProjectionSubgroup("edge", None)
            self.assertEqual((None, None, True), config0.getGroupProjectionSubgroup("edge"))
            # must re-run initial configuration to use projection subgroup
            config0.run()

            align1 = FitterStepAlign()
            fitter.addFitterStep(align1)
            align1.setAlignManually(True)
            align1.setRotation([0.0, 0.0, 0.0])
            align1.setScale(1.179)
            align1.setTranslation([0.251, 0.023, -0.01147])
            align1.run()

            fit2 = FitterStepFit()
            fitter.addFitterStep(fit2)
            fit2.setGroupStrainPenalty(None, [0.01])
            fit2.setGroupCurvaturePenalty(None, [1.0])
            fit2.setNumberOfIterations(2)
            fit2.run()

            config3 = FitterStepConfig()
            fitter.addFitterStep(config3)
            config3.setGroupOutlierLength("trunk", -0.5)
            # switch subgroup centroid over to trunk group, which is equivalent
            self.assertEqual((centroid, False, True), config3.getGroupProjectionSubgroup(None))
            config3.setGroupProjectionSubgroup(None, None)
            self.assertEqual((None, None, True), config3.getGroupProjectionSubgroup(None))
            config3.clearGroupProjectionSubgroup(None)
            self.assertEqual((centroid, False, True), config3.getGroupProjectionSubgroup(None))
            config3.setGroupProjectionSubgroup(None, None)
            self.assertEqual((None, None, True), config3.getGroupProjectionSubgroup(None))
            self.assertEqual((None, False, True), config3.getGroupProjectionSubgroup("trunk"))
            config3.setGroupProjectionSubgroup("trunk", centroid)
            self.assertEqual((centroid, True, True), config3.getGroupProjectionSubgroup("trunk"))
            config3.run()

            fit4 = FitterStepFit()
            fitter.addFitterStep(fit4)
            fit4.setNumberOfIterations(2)
            fit4.run()

            TOL = 1.0E-7
            rmsError, maxError = fitter.getDataRMSAndMaximumProjectionError()
            minElementIdentifier, minJacobian = fitter.getLowestElementJacobian()
            self.assertAlmostEqual(rmsError, 0.01788062932269768, delta=TOL)
            self.assertAlmostEqual(maxError, 0.0660542949294517, delta=TOL)
            self.assertEqual(minElementIdentifier, 2)
            self.assertAlmostEqual(minJacobian, 1.165478981129574, delta=TOL)

            # check all projected coordinates are to the centroid of the trunk
            fieldcache = fieldmodule.createFieldcache()
            activeDataNodeset = fitter.getActiveDataNodesetGroup()
            dataHostLocation = fitter.getDataHostLocationField()
            edge = fieldmodule.findFieldByName("edge").castGroup()
            self.assertTrue(edge.isValid())
            edgeData = edge.getNodesetGroup(fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS))
            nodeiter = activeDataNodeset.createNodeiterator()
            node = nodeiter.next()
            while node.isValid():
                fieldcache.setNode(node)
                element, xi = dataHostLocation.evaluateMeshLocation(fieldcache, 3)
                self.assertTrue(element.getIdentifier() in [1, 2])
                self.assertTrue(0.0 <= xi[0] <= 1.0)
                if edgeData.containsNode(node):
                    self.assertAlmostEqual(xi[1], 0.0, delta=TOL)
                    self.assertAlmostEqual(xi[2], 1.0, delta=TOL)
                else:
                    self.assertTrue(0.0 <= xi[1] <= 1.0)
                    self.assertAlmostEqual(xi[2], 0.5, delta=TOL)
                node = nodeiter.next()

            if i == 1:
                # build more of the model and fit it to branch1 data
                # transform the same nerve box model to align with branch1 and add it as new elements and nodes
                tmpRegion = region.createRegion()
                tmpRegion.readFile(zinc_model_file_name)
                tmpFieldmodule = tmpRegion.getFieldmodule()
                with ChangeManager(tmpFieldmodule):
                    tmpCoordinates = tmpFieldmodule.findFieldByName("coordinates").castFiniteElement()
                    newCoordinates = tmpFieldmodule.createFieldMatrixMultiply(
                        3, tmpFieldmodule.createFieldConstant([0.0, -1.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 1.0]),
                        tmpCoordinates) + tmpFieldmodule.createFieldConstant([1.0, 0.0, 0.0])
                    fieldassignment = tmpCoordinates.createFieldassignment(newCoordinates)
                    fieldassignment.assign()
                    # rename trunk -> branch1
                    tmpGroup = tmpFieldmodule.findFieldByName("trunk").castGroup()
                    tmpGroup.setName("branch1")
                    # offset element, face, line and node numbers to not clash with trunk
                    identifierOffset = 100
                    maxObjectIdentifiers = [3, 42, 19, 2]
                    for dimension in range(3, 0, -1):
                        mesh = tmpFieldmodule.findMeshByDimension(dimension)
                        for elementIdentifier in range(1, 101):
                            element = mesh.findElementByIdentifier(elementIdentifier)
                            if not element.isValid():
                                self.assertEqual(elementIdentifier, maxObjectIdentifiers[dimension] + 1)
                                break
                            element.setIdentifier(elementIdentifier + identifierOffset)
                    nodeset = tmpFieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
                    for nodeIdentifier in range(1, 101):
                        node = nodeset.findNodeByIdentifier(nodeIdentifier)
                        if not node.isValid():
                            self.assertEqual(nodeIdentifier, maxObjectIdentifiers[0] + 1)
                            break
                        node.setIdentifier(nodeIdentifier + identifierOffset)
                    # read into main region via in-memory EX file buffer
                    buffer = write_to_buffer(tmpRegion, resource_domain_type=
                        Field.DOMAIN_TYPE_NODES | Field.DOMAIN_TYPE_MESH1D |
                        Field.DOMAIN_TYPE_MESH2D | Field.DOMAIN_TYPE_MESH3D)
                    self.assertTrue(buffer is not None)
                    self.assertEqual(read_from_buffer(region, buffer), RESULT_OK)

                # move the edge data point to work for branch1 so solution is not singular
                edgeGroup = fieldmodule.findFieldByName("edge").castGroup()
                datapoints = fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
                edgeNodesetGroup = edgeGroup.getNodesetGroup(datapoints)
                node = edgeNodesetGroup.createNodeiterator().next()
                fieldcache = fieldmodule.createFieldcache()
                fieldcache.setNode(node)
                self.assertEqual(coordinates.assignReal(fieldcache, [1.0, 0.2, 0.095]), RESULT_OK)

                # fit the branch with 1 new fit step
                fitterb = Fitter(region=region)
                fitterb.setDiagnosticLevel(1)
                branch1Group = fieldmodule.findFieldByName("branch1").castGroup()
                fitterb.setModelCoordinatesField(coordinates)
                fitterb.setModelFitGroup(branch1Group)
                fitterb.defineCommonMeshFields()

                config0b = fitterb.getInitialFitterStepConfig()
                config0b.setGroupProjectionSubgroup("branch1", centroid)
                self.assertEqual((centroid, True, False), config0b.getGroupProjectionSubgroup("branch1"))
                fitterb.setDataCoordinatesField(coordinates)
                markerGroup = fieldmodule.findFieldByName("marker").castGroup()
                if markerGroup.isValid():
                    fitterb.setMarkerGroup(markerGroup)
                fitterb.defineDataProjectionFields()
                fitterb.initializeFit()

                fit1b = FitterStepFit()
                fitterb.addFitterStep(fit1b)
                fit1b.setGroupStrainPenalty(None, [0.1])
                fit1b.setGroupCurvaturePenalty(None, [0.1])
                fit1b.setNumberOfIterations(3)
                fit1b.run()

                rmsError, maxError = fitterb.getDataRMSAndMaximumProjectionError()
                minElementIdentifier, minJacobian = fitterb.getLowestElementJacobian()
                self.assertAlmostEqual(rmsError, 0.02347727286951435, delta=TOL)
                self.assertAlmostEqual(maxError, 0.04547098724783385, delta=TOL)
                self.assertEqual(minElementIdentifier, 102)
                self.assertAlmostEqual(minJacobian, 0.8073661446227143, delta=TOL)

if __name__ == "__main__":
    unittest.main()
