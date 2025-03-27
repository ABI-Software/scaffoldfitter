import logging
import os
import sys
import unittest
from cmlibs.utils.zinc.field import createFieldMeshIntegral
from cmlibs.zinc.result import RESULT_OK
from scaffoldfitter.fitter import Fitter
from scaffoldfitter.fitterstepfit import FitterStepFit

here = os.path.abspath(os.path.dirname(__file__))


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Fit2dTestCase(unittest.TestCase):

    def test_fit_breast2d(self):
        """
        Test 2D fit with curvature penalty requiring fibre field to be set.
        """
        zinc_model_file = os.path.join(here, "resources", "breast_plate.exf")
        zinc_data_file = os.path.join(here, "resources", "breast_data.exf")
        fitter = Fitter(zinc_model_file, zinc_data_file)
        fitter.setDiagnosticLevel(1)
        fitter.load()

        fit1 = FitterStepFit()
        fitter.addFitterStep(fit1)
        self.assertEqual(2, len(fitter.getFitterSteps()))
        fit1.setGroupCurvaturePenalty(None, [100.0])
        # this example requires a non-zero data sliding factor for stability as just a flat surface
        fit1.setGroupDataSlidingFactor(None, 0.01)
        self.assertEqual((0.01, True, False), fit1.getGroupDataSlidingFactor(None))
        # can't use a curvature penalty without a fibre field
        with self.assertRaises(AssertionError) as cm:
            fit1.run()
        self.assertEqual(str(cm.exception),
                         "Must supply a fibre field to use strain/curvature penalties "
                         "with mesh dimension < coordinate components.")

        # set the in-built zero fibres field
        fieldmodule = fitter.getFieldmodule()
        zeroFibreField = fieldmodule.findFieldByName("zero fibres")
        self.assertTrue(zeroFibreField.isValid())
        fitter.setFibreField(zeroFibreField)
        fitter.load()

        # check these now as different after re-load
        fieldmodule = fitter.getFieldmodule()
        coordinates = fitter.getModelCoordinatesField()
        self.assertEqual(coordinates.getName(), "coordinates")
        self.assertEqual(fitter.getDataCoordinatesField().getName(), "data_coordinates")

        fit1.run()

        # check surface area of fitted coordinates
        # Note name is only prefixes with "fitted " when written with Fitter.writeModel
        surfaceAreaField = createFieldMeshIntegral(coordinates, fitter.getMesh(2), number_of_points=4)
        valid = surfaceAreaField.isValid()
        self.assertTrue(surfaceAreaField.isValid())
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 104501.36293993103, delta=1.0E-1)

    def test_projection_error(self):
        """
        Test data projection RMS and maximum error calculations.
        """
        zinc_model_file = os.path.join(here, "resources", "square.exf")
        zinc_data_file = os.path.join(here, "resources", "square_error_data.exf")
        fitter = Fitter(zinc_model_file, zinc_data_file)
        fitter.setDiagnosticLevel(1)
        fitter.load()
        rmsErrorValue, maxErrorValue = fitter.getDataRMSAndMaximumProjectionError()
        TOL = 1.0E-10
        self.assertAlmostEqual(rmsErrorValue, 0.34641016151377546, delta=TOL)  # sqrt(0.12)
        self.assertAlmostEqual(maxErrorValue, 0.5, delta=TOL)


if __name__ == "__main__":
    unittest.main()
