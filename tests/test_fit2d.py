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
        fit1.run()
        self.assertFalse(fit1.hasRun())

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
        self.assertTrue(fit1.hasRun())

        # check surface area of fitted coordinates
        # Note name is only prefixes with "fitted " when written with Fitter.writeModel
        surfaceAreaField = createFieldMeshIntegral(coordinates, fitter.getMesh(2), number_of_points=4)
        valid = surfaceAreaField.isValid()
        self.assertTrue(surfaceAreaField.isValid())
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertEqual(result, RESULT_OK)
        self.assertAlmostEqual(surfaceArea, 104501.36293993103, delta=1.0E-1)


    def test_fit_stretch_house2d(self):
        """
        Test 2D fit of house consisting of 4 biquadratic Lagrange elements in two groups,
        "left" and "right" with 2 biquadratic simplex elements making up the "attic" group.
             *
          / A|A \
        *----*----*
        |  L | R  |
        *----*----*
        |  L | R  |
        *----*----*

        This is fitted to data which blows this shape out to being on a larger sphere. It
        tests both stretching of the 2-D surface and whether boundary points are strictly on
        the boundary of groups; not C0 edges in the interior.
        """
        zinc_model_file = os.path.join(here, "resources", "house2d.exf")
        zinc_data_file = os.path.join(here, "resources", "house2d_data.exf")
        fitter = Fitter(zinc_model_file, zinc_data_file)
        fitter.setDiagnosticLevel(0)

        fitter.load()

        fieldmodule = fitter.getFieldmodule()
        coordinates = fitter.getModelCoordinatesField()
        self.assertEqual(coordinates.getName(), "coordinates")
        self.assertEqual(fitter.getDataCoordinatesField().getName(), "coordinates")
        # set the in-built zero fibres field for strain and curvature on 2-D mesh, needed before fit steps
        zeroFibreField = fieldmodule.findFieldByName("zero fibres")
        self.assertTrue(zeroFibreField.isValid())
        fitter.setFibreField(zeroFibreField)

        fit1 = FitterStepFit()
        fitter.addFitterStep(fit1)
        self.assertEqual(2, len(fitter.getFitterSteps()))
        fit1.setGroupDataSlidingFactor(None, 0.01)
        fit1.setGroupDataStretchFactor(None, 5.0)
        self.assertEqual((5.0, True, False), fit1.getGroupDataStretchFactor(None))
        fit1.setGroupStrainPenalty(None, [0.01])
        fit1.setGroupCurvaturePenalty(None, [1.0])
        fit1.run()
        self.assertTrue(fit1.hasRun())

        TOL = 2.0E-8
        # check surface area, and erros of fitted coordinates
        surfaceAreaField = createFieldMeshIntegral(coordinates, fitter.getMesh(2), number_of_points=4)
        self.assertTrue(surfaceAreaField.isValid())
        fieldcache = fieldmodule.createFieldcache()
        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertAlmostEqual(surfaceArea, 34.13089457038114, delta=TOL)
        rmsError, maxError = fitter.getDataRMSAndMaximumProjectionError()
        self.assertAlmostEqual(rmsError, 0.22124894917663576, delta=TOL)
        self.assertAlmostEqual(maxError, 0.75486586220124236, delta=TOL)

        fit2 = FitterStepFit()
        fitter.addFitterStep(fit2)
        self.assertEqual(3, len(fitter.getFitterSteps()))
        fit2.setGroupCurvaturePenalty(None, [0.1])
        fit2.run()
        self.assertTrue(fit2.hasRun())

        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertAlmostEqual(surfaceArea, 39.120692505119095, delta=10.0 * TOL)
        rmsError, maxError = fitter.getDataRMSAndMaximumProjectionError()
        self.assertAlmostEqual(rmsError, 0.07553286449754204, delta=10.0 * TOL)
        self.assertAlmostEqual(maxError, 0.4123847154550016, delta=10.0 * TOL)

        fit3 = FitterStepFit()
        fitter.addFitterStep(fit3)
        self.assertEqual(4, len(fitter.getFitterSteps()))
        fit3.run()
        self.assertTrue(fit3.hasRun())

        result, surfaceArea = surfaceAreaField.evaluateReal(fieldcache, 1)
        self.assertAlmostEqual(surfaceArea, 42.5708819901009, delta=100.0 * TOL)
        rmsError, maxError = fitter.getDataRMSAndMaximumProjectionError()
        self.assertAlmostEqual(rmsError, 0.031508815126039924, delta=100.0 * TOL)
        self.assertAlmostEqual(maxError, 0.22486323403558317, delta=100.0 * TOL)

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
