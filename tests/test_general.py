import math
import os
import unittest
from cmlibs.utils.zinc.field import createFieldMeshIntegral
from cmlibs.zinc.result import RESULT_OK
from scaffoldfitter.fitter import Fitter
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


if __name__ == "__main__":
    unittest.main()
