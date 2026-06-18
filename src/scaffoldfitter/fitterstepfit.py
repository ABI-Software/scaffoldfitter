"""
Fit step for gross alignment and scale.
"""

from cmlibs.utils.zinc.general import ChangeManager
from cmlibs.zinc.optimisation import Optimisation
from cmlibs.zinc.result import RESULT_OK
from scaffoldfitter.fitterstep import FitterStep
import logging
import sys


logger = logging.getLogger(__name__)


class FitterStepFit(FitterStep):

    _jsonTypeId = "_FitterStepFit"
    _dataWeightToken = "dataWeight"
    _dataSlidingFactorToken = "dataSlidingFactor"
    _dataStretchToken = "dataStretch"
    _strainPenaltyToken = "strainPenalty"
    _curvaturePenaltyToken = "curvaturePenalty"

    def __init__(self):
        super(FitterStepFit, self).__init__()
        self._numberOfIterations = 1
        self._maximumSubIterations = 1
        self._updateReferenceState = False

    @classmethod
    def getJsonTypeId(cls):
        return cls._jsonTypeId

    def decodeSettingsJSONDict(self, dctIn: dict):
        """
        Decode definition of step from JSON dict.
        """
        super().decodeSettingsJSONDict(dctIn)  # to decode group settings
        # ensure all new options are in dct
        dct = self.encodeSettingsJSONDict()
        dct.update(dctIn)
        self._numberOfIterations = dct["numberOfIterations"]
        self._maximumSubIterations = dct["maximumSubIterations"]
        self._updateReferenceState = dct["updateReferenceState"]

    def encodeSettingsJSONDict(self) -> dict:
        """
        Encode definition of step in dict.
        :return: Settings in a dict ready for passing to json.dump.
        """
        dct = super().encodeSettingsJSONDict()
        dct.update({
            "numberOfIterations": self._numberOfIterations,
            "maximumSubIterations": self._maximumSubIterations,
            "updateReferenceState": self._updateReferenceState
            })
        return dct

    def clearGroupDataWeight(self, groupName):
        """
        Clear group data weight so fall back to last fit or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._dataWeightToken)

    def getGroupDataWeight(self, groupName):
        """
        Get group data weight to apply in fit, and associated flags.
        If not set or inherited, gets value from default group.
        :param groupName:  Exact model group name, or None for default group.
        :return: Weight, setLocally, inheritable.
        Weight is a real value >= 0.0. Default value 1.0 if not set.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        return self.getGroupSetting(groupName, self._dataWeightToken, 1.0)

    def setGroupDataWeight(self, groupName, weight):
        """
        Set group data weight to apply in fit, or reset to use default.
        :param groupName:  Exact model group name, or None for default group.
        :param weight:  Float valued weight >= 0.0, or None to reset to global
        default. Function ensures value is valid.
        Weight is a real value multiplying the data projection error. Higher
        values for a group make the fit more closely match its data relative to
        other groups. It is recommended that the default group data weight be
        kept at 1.0 and other weights or penalties changed relative to it.
        """
        if weight is not None:
            if not isinstance(weight, float):
                return
            if weight < 0.0:
                weight = 0.0
        self.setGroupSetting(groupName, self._dataWeightToken, weight)

    def clearGroupDataSlidingFactor(self, groupName):
        """
        Clear group data sliding factor so fall back to last fit or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._dataSlidingFactorToken)

    def getGroupDataSlidingFactor(self, groupName):
        """
        Get group data sliding factor to apply in fit, and associated flags.
        If not set or inherited, gets value from default group.
        :param groupName:  Exact model group name, or None for default group.
        :return: Sliding factor, setLocally, inheritable.
        Sliding factor is a real value >= 0.0 which multiplies group weight in
        sliding directions.
        Default value 0.1 gives some sliding resistance.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        return self.getGroupSetting(groupName, self._dataSlidingFactorToken, 0.1)

    def setGroupDataSlidingFactor(self, groupName, slidingFactor):
        """
        Set group data sliding weight to apply in fit, or reset to use default.
        :param groupName:  Exact model group name, or None for default group.
        :param slidingFactor:  Float value >= 0.0, or None to reset to global
        default. Function ensures value is valid.
        Sliding factor is a real value >= 0.0 which multiplies group weight in
        sliding directions (two directions for surfaces, one for lines).
        Default value 0.1 gives some sliding resistance.
        Setting value 0.0 gives zero sliding resistance.
        A small positive value << 1.0 may aid stability where there is
        insufficient constraint from markers, line groups, multiple groups
        and inherent shape.
        Higher values increasingly apply stretch to the span of data points,
        but also limit movement which can lead to tangential wrinkling.
        """
        if slidingFactor is not None:
            if not isinstance(slidingFactor, float):
                return
            if slidingFactor < 0.0:
                slidingFactor = 0.0
        self.setGroupSetting(groupName, self._dataSlidingFactorToken, slidingFactor)

    def clearGroupDataStretch(self, groupName):
        """
        Clear local group data stretch so fall back to last config or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._dataStretchToken)

    def getGroupDataStretch(self, groupName):
        """
        Get flag controlling whether tangential projections have the full
        data weight applied to them to stretch the model to the span of data
        with zero/low sliding factor. Default is True/on.
        :param groupName:  Exact model group name, or None for default group.
        :return:  Data stretch flag, setLocally, inheritable.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        return self.getGroupSetting(groupName, self._dataStretchToken, True)

    def setGroupDataStretch(self, groupName, dataStretch):
        """
        Set flag controlling whether tangential projections have the full
        data weight applied to them to stretch the model to the span of data
        with zero/low sliding factor. Turn off for groups such as inlet/outlet
        tubes where specimens show quite variable length, so feature is oriented
        with the data but keeps its length from the reference scaffold.
        :param groupName:  Exact model group name, or None for default group.
        :param dataStretch:  Boolean True/False or None to reset to global
        default. Function ensures value is valid.
        """
        if dataStretch is not None:
            if not isinstance(dataStretch, bool):
                return
        self.setGroupSetting(groupName, self._dataStretchToken, dataStretch)

    def clearGroupStrainPenalty(self, groupName: str):
        """
        Clear local group strain penalty so fall back to last fit or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._strainPenaltyToken)

    def getGroupStrainPenalty(self, groupName, count=None):
        """
        Get list of strain penalty factors used to scale first deformation
        gradient components in group. Up to 9 components possible in 3-D, 4 in 2-D.
        :param groupName:  Exact model group name, or None for default group.
        :param count: Optional number of factors to limit or enlarge list to.
        If enlarging, values are padded with the last stored value. If None,
        the number stored is requested.
        If not set or inherited, gets value from default group.
        :return: list(float), setLocally, inheritable.
        First return value is a list of float strain penalty factors, length > 0.
        If length is 1 and value is 0.0, no penalty will be applied.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        strainPenalty, setLocally, inheritable = self.getGroupSetting(groupName, self._strainPenaltyToken, [0.0])
        if count:
            count = min(count, 9)
            storedCount = len(strainPenalty)
            if count <= storedCount:
                strainPenalty = strainPenalty[:count]
            else:
                lastFactor = strainPenalty[-1]
                strainPenalty = strainPenalty + [lastFactor]*(count - storedCount)
        else:
            strainPenalty = strainPenalty[:]  # shallow copy
        return strainPenalty, setLocally, inheritable

    def setGroupStrainPenalty(self, groupName, strainPenalty):
        """
        :param groupName:  Exact model group name, or None for default group.
        :param strainPenalty: List of 1-9 float-value strain penalty factors to scale
        first deformation gradient components, or None to reset to inherited or
        default value. If fewer than 9 values are supplied in the list, the
        last value is used for all remaining components.
        """
        if strainPenalty is not None:
            assert isinstance(strainPenalty, list), "FitterStepFit: setGroupStrainPenalty requires a list of float"
            strainPenalty = strainPenalty[:9]  # shallow copy, limiting size
            count = len(strainPenalty)
            assert count > 0, "FitterStepFit: setGroupStrainPenalty requires a list of at least 1 float"
            for i in range(count):
                assert isinstance(strainPenalty[i], float), \
                    "FitterStepFit: setGroupStrainPenalty requires a list of float"
                if strainPenalty[i] < 0.0:
                    strainPenalty[i] = 0.0
        self.setGroupSetting(groupName, self._strainPenaltyToken, strainPenalty)

    def clearGroupCurvaturePenalty(self, groupName):
        """
        Clear local group curvature penalty so fall back to last fit or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._curvaturePenaltyToken)

    def getGroupCurvaturePenalty(self, groupName, count=None):
        """
        Get list of curvature penalty factors used to scale second deformation
        gradient components in group. Up to 27 components possible in 3-D.
        :param groupName:  Exact model group name, or None for default group.
        :param count: Optional number of factors to limit or enlarge list to.
        If enlarging, values are padded with the last stored value. If None,
        the number stored is requested.
        If not set or inherited, gets value from default group.
        :return: list(float), setLocally, inheritable.
        First return value is a list of float curvature penalty factors.
        If length is 1 and value is 0.0, no penalty will be applied.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        curvaturePenalty, setLocally, inheritable = self.getGroupSetting(groupName, self._curvaturePenaltyToken, [0.0])
        if count:
            storedCount = len(curvaturePenalty)
            if count <= storedCount:
                curvaturePenalty = curvaturePenalty[:count]
            else:
                lastFactor = curvaturePenalty[-1]
                curvaturePenalty = curvaturePenalty + [lastFactor]*(count - storedCount)
        else:
            curvaturePenalty = curvaturePenalty[:]  # shallow copy
        return curvaturePenalty, setLocally, inheritable

    def setGroupCurvaturePenalty(self, groupName, curvaturePenalty):
        """
        :param groupName:  Exact model group name, or None for default group.
        :param curvaturePenalty: List of 1-27 float-value curvature penalty
        factors to scale first deformation gradient components, or None to
        reset to inherited or default value. If fewer than 27 values are
        supplied in the list, the last value is used for all remaining
        components.
        """
        if curvaturePenalty is not None:
            assert isinstance(curvaturePenalty, list), \
                "FitterStepFit: setGroupCurvaturePenalty requires a list of float"
            curvaturePenalty = curvaturePenalty[:27]  # shallow copy, limiting size
            count = len(curvaturePenalty)
            assert count > 0, "FitterStepFit: setGroupCurvaturePenalty requires a list of at least 1 float"
            for i in range(count):
                assert isinstance(curvaturePenalty[i], float), \
                    "FitterStepFit: setGroupCurvaturePenalty requires a list of float"
                if curvaturePenalty[i] < 0.0:
                    curvaturePenalty[i] = 0.0
        self.setGroupSetting(groupName, self._curvaturePenaltyToken, curvaturePenalty)

    def getNumberOfIterations(self):
        return self._numberOfIterations

    def setNumberOfIterations(self, numberOfIterations):
        assert numberOfIterations > 0
        if numberOfIterations != self._numberOfIterations:
            self._numberOfIterations = numberOfIterations
            return True
        return False

    def getMaximumSubIterations(self):
        return self._maximumSubIterations

    def setMaximumSubIterations(self, maximumSubIterations):
        assert maximumSubIterations > 0
        if maximumSubIterations != self._maximumSubIterations:
            self._maximumSubIterations = maximumSubIterations
            return True
        return False

    def isUpdateReferenceState(self):
        return self._updateReferenceState

    def setUpdateReferenceState(self, updateReferenceState):
        if updateReferenceState != self._updateReferenceState:
            self._updateReferenceState = updateReferenceState
            return True
        return False

    def run(self, modelFileNameStem=None):
        """
        Fit model geometry parameters to data.
        :param modelFileNameStem: Optional name stem of intermediate output file to write.
        """
        self._fitter.assignDataWeights(self)
        deformActiveMeshGroup, strainActiveMeshGroup, curvatureActiveMeshGroup = \
            self._fitter.assignDeformationPenalties(self)

        fieldmodule = self._fitter.getFieldmodule()
        optimisation = fieldmodule.createOptimisation()
        optimisation.setMethod(Optimisation.METHOD_NEWTON)
        optimisation.addDependentField(self._fitter.getModelCoordinatesField())
        if self._fitter.getModelFitGroup():
            optimisation.setConditionalField(self._fitter.getModelCoordinatesField(), self._fitter.getModelFitGroup())
        optimisation.setAttributeInteger(Optimisation.ATTRIBUTE_MAXIMUM_ITERATIONS, self._maximumSubIterations)

        deformationPenaltyObjective = None
        with ChangeManager(fieldmodule):
            dataObjective = self.createDataObjectiveField()
            result = optimisation.addObjectiveField(dataObjective)
            assert result == RESULT_OK, "Fit Geometry:  Could not add data objective field"
            deformationPenaltyObjective = self.createDeformationPenaltyObjectiveField(
                deformActiveMeshGroup, strainActiveMeshGroup, curvatureActiveMeshGroup)
            if deformationPenaltyObjective:
                result = optimisation.addObjectiveField(deformationPenaltyObjective)
                assert result == RESULT_OK, "Fit Geometry:  Could not add strain/curvature penalty objective field"
            flattenGroupObjective = self.createFlattenGroupObjectiveField()
            if flattenGroupObjective:
                result = optimisation.addObjectiveField(flattenGroupObjective)
                assert result == RESULT_OK, "Fit Geometry:  Could not add flatten group objective field"

        fieldcache = fieldmodule.createFieldcache()
        objectiveFormat = "{:12e}"
        success = True
        for iterationIndex in range(self._numberOfIterations):
            iterName = str(iterationIndex + 1)
            if self.getDiagnosticLevel() > 0:
                print("-------- Iteration " + iterName)
            if self.getDiagnosticLevel() > 0:
                result, objective = dataObjective.evaluateReal(fieldcache, 1)
                print("    Data objective", objectiveFormat.format(objective))
                if deformationPenaltyObjective:
                    result, objective = deformationPenaltyObjective.evaluateReal(
                        fieldcache, deformationPenaltyObjective.getNumberOfComponents())
                    print("    Deformation penalty objective", objectiveFormat.format(objective))
                if flattenGroupObjective:
                    result, objective = flattenGroupObjective.evaluateReal(
                        fieldcache, flattenGroupObjective.getNumberOfComponents())
                    print("    Flatten group objective", objectiveFormat.format(objective))
            result = optimisation.optimise()
            if self.getDiagnosticLevel() > 1:
                solutionReport = optimisation.getSolutionReport()
                print(solutionReport)
            if result != RESULT_OK:
                logger.error("Fit Geometry:  Optimisation failed with result " + str(result))
                success = False
                # self._fitter.print_log()
                break
            self._fitter.calculateDataProjections(self)
            if modelFileNameStem:
                self._fitter.writeModel(modelFileNameStem + "_fit" + iterName + ".exf")

        if self.getDiagnosticLevel() > 0:
            print("--------")
            result, objective = dataObjective.evaluateReal(fieldcache, 1)
            print("    END Data objective", objectiveFormat.format(objective))
            if deformationPenaltyObjective:
                result, objective = deformationPenaltyObjective.evaluateReal(
                    fieldcache, deformationPenaltyObjective.getNumberOfComponents())
                print("    END Deformation penalty objective", objectiveFormat.format(objective))
            if flattenGroupObjective:
                result, objective = flattenGroupObjective.evaluateReal(
                    fieldcache, flattenGroupObjective.getNumberOfComponents())
                print("    Flatten group objective", objectiveFormat.format(objective))
            if self.getDiagnosticLevel() > 1:
                self._fitter.print_log()

        if self._updateReferenceState:
            self._fitter.updateModelReferenceCoordinates()

        self.setHasRun(success)

    def createDataObjectiveField(self):
        """
        Get FieldNodesetSum objective for data projected onto mesh, including markers with fixed locations.
        Assumes ChangeManager(fieldmodule) is in effect.
        :return: Zinc FieldNodesetSum.
        """
        fieldmodule = self._fitter.getFieldmodule()
        delta = self._fitter.getDataDeltaField()
        weight = self._fitter.getDataWeightField()
        # non-oriented data for full n-way constraints
        # deltaSq = fieldmodule.createFieldDotProduct(delta, delta)
        # oriented delta aligns with tangents and normals to mesh groups so can set separate
        # sliding constraints for line/surface projections, full constraint for marker points
        dataProjectionOrientation = self._fitter.getDataProjectionOrientationField()
        orientedDelta = fieldmodule.createFieldMatrixMultiply(
            delta.getNumberOfComponents(), dataProjectionOrientation, delta)
        deltaSq = fieldmodule.createFieldMultiply(orientedDelta, orientedDelta)
        weightedDeltaSq = fieldmodule.createFieldDotProduct(weight, deltaSq)
        dataProjectionObjective = fieldmodule.createFieldNodesetSum(
            weightedDeltaSq, self._fitter.getActiveDataNodesetGroup())
        dataProjectionObjective.setElementMapField(self._fitter.getDataHostLocationField())
        return dataProjectionObjective

    def createDeformationPenaltyObjectiveField(self, deformActiveMeshGroup, strainActiveMeshGroup,
                                               curvatureActiveMeshGroup):
        """
        Get strain and curvature penalty mesh integral objective field.
        Assumes ChangeManager(fieldmodule) is in effect.
        :param deformActiveMeshGroup: Mesh group over which either penalties is applied.
        :param strainActiveMeshGroup: Mesh group over which strain penalty is applied.
        :param curvatureActiveMeshGroup: Mesh group over which curvature penalty is applied.
        :return: Zinc FieldMeshIntegral, or None if not applied.
        """
        if deformActiveMeshGroup.getSize() == 0:
            return None
        applyStrainPenalty = strainActiveMeshGroup.getSize() > 0
        applyCurvaturePenalty = curvatureActiveMeshGroup.getSize() > 0
        if not (applyStrainPenalty or applyCurvaturePenalty):
            return None
        numberOfGaussPoints = 3
        fieldmodule = self._fitter.getFieldmodule()
        mesh = self._fitter.getHighestDimensionMesh()
        modelCoordinates = self._fitter.getModelCoordinatesField()
        modelReferenceCoordinates = self._fitter.getModelReferenceCoordinatesField()
        fibreField = self._fitter.getFibreField()
        dimension = mesh.getDimension()
        coordinatesCount = modelCoordinates.getNumberOfComponents()
        assert (coordinatesCount == dimension) or fibreField, \
            "Must supply a fibre field to use strain/curvature penalties with mesh dimension < coordinate components."
        deformationGradient1 = deformationGradient1raw = fieldmodule.createFieldGradient(
            modelCoordinates, modelReferenceCoordinates)
        fibreAxes = None
        fibreAxesT = None
        if fibreField:
            # convert to local fibre directions, with possible dimension reduction for 2D, 1D
            fibreAxes = fieldmodule.createFieldFibreAxes(fibreField, modelReferenceCoordinates)
            if not fibreAxes.isValid():
                self.getFitter().print_log()
            if dimension == 3:
                fibreAxesT = fieldmodule.createFieldTranspose(3, fibreAxes)
            elif dimension == 2:
                fibreAxesT = fieldmodule.createFieldComponent(
                    fibreAxes, [1, 4, 2, 5, 3, 6] if (coordinatesCount == 3) else [1, 4, 2, 5])
            else:  # dimension == 1
                fibreAxesT = fieldmodule.createFieldComponent(
                    fibreAxes, [1, 2, 3] if (coordinatesCount == 3) else [1, 2] if (coordinatesCount == 2) else [1])
        deformationTerm = None
        if applyStrainPenalty:
            # large strain
            if fibreField:
                deformationGradient1 = fieldmodule.createFieldMatrixMultiply(
                    coordinatesCount, deformationGradient1raw, fibreAxesT)
            deformationGradient1T = fieldmodule.createFieldTranspose(coordinatesCount, deformationGradient1)
            C = fieldmodule.createFieldMatrixMultiply(dimension, deformationGradient1T, deformationGradient1)
            alpha = self._fitter.getStrainPenaltyField()
            I = fieldmodule.createFieldConstant(
                [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0] if (dimension == 3) else
                [1.0, 0.0, 0.0, 1.0] if (dimension == 2) else
                [1.0])
            E2 = C - I
            wtSqE2 = fieldmodule.createFieldDotProduct(alpha, E2 * E2)
            deformationTerm = wtSqE2
        if applyCurvaturePenalty:
            # second order Sobolev smoothing terms
            # don't do gradient of deformationGradient1 with fibres due to slow finite difference evaluation
            deformationGradient2 = fieldmodule.createFieldGradient(deformationGradient1raw, modelReferenceCoordinates)
            if fibreField:
                # convert to local fibre directions
                deformationGradient2a = fieldmodule.createFieldMatrixMultiply(
                    coordinatesCount*coordinatesCount, deformationGradient2, fibreAxesT)
                # transpose each deformation component of deformationGradient2a to remultiply by fibreAxesT
                if dimension == 1:
                    deformationGradient2aT = deformationGradient2a
                else:
                    transposeComponents = None
                    if coordinatesCount == 3:
                        if dimension == 3:
                            transposeComponents = [1, 4, 7, 2, 5, 8, 3, 6, 9,
                                                   10, 13, 16, 11, 14, 17, 12, 15, 18,
                                                   19, 22, 25, 20, 23, 26, 21, 24, 27]
                        elif dimension == 2:
                            transposeComponents = [1, 3, 5, 2, 4, 6, 7, 9, 11, 8, 10, 12, 13, 15, 17, 14, 16, 18]
                    elif coordinatesCount == 2:
                        transposeComponents = [1, 3, 2, 4, 5, 7, 6, 8]
                    deformationGradient2aT = \
                        fieldmodule.createFieldComponent(deformationGradient2a, transposeComponents)
                deformationGradient2 = fieldmodule.createFieldMatrixMultiply(
                    dimension*coordinatesCount, deformationGradient2aT, fibreAxesT)
            beta = self._fitter.getCurvaturePenaltyField()
            wtSqDeformationGradient2 = \
                fieldmodule.createFieldDotProduct(beta, deformationGradient2*deformationGradient2)
            deformationTerm = \
                (deformationTerm + wtSqDeformationGradient2) if deformationTerm else wtSqDeformationGradient2
            if not deformationTerm.isValid():
                self.getFitter().print_log()
                raise AssertionError("Scaffoldfitter: Failed to get deformation term")

        deformationPenaltyObjective = fieldmodule.createFieldMeshIntegral(
            deformationTerm, self._fitter.getModelReferenceCoordinatesField(), deformActiveMeshGroup)
        deformationPenaltyObjective.setNumbersOfPoints(numberOfGaussPoints)
        return deformationPenaltyObjective

    def createFlattenGroupObjectiveField(self):
        """
        Get flatten group penalty mesh integral field, if any.
        Assumes ChangeManager(fieldmodule) is in effect.
        :return: Zinc FieldMeshIntegral, or None if not applied.
        """
        flattenGroup = self._fitter.getFlattenGroup()
        if not flattenGroup:
            return None
        flattenGroupName = flattenGroup.getName()
        flattenMeshGroup = None
        for dimension in range(self._fitter.getHighestDimensionMesh().getDimension(), 0, -1):
            mesh = self._fitter.getMesh(dimension)
            flattenMeshGroup = flattenGroup.getMeshGroup(mesh)
            if flattenMeshGroup.isValid() and (flattenMeshGroup.getSize() > 0):
                break
        else:
            if self.getDiagnosticLevel() > 0:
                print("Flatten group " + flattenGroupName + " is empty")
            return None
        weight = self.getGroupDataWeight(flattenGroupName)[0]
        if weight <= 0.0:
            if self.getDiagnosticLevel() > 0:
                print("Flatten group " + flattenGroupName + " has zero weight")
            return None

        fieldmodule = self._fitter.getFieldmodule()
        modelCoordinates = self._fitter.getModelCoordinatesField()
        flattenComponent = fieldmodule.createFieldComponent(modelCoordinates, modelCoordinates.getNumberOfComponents())
        flattenWeight = fieldmodule.createFieldConstant([weight])
        flattenComponentWeighted = flattenWeight * flattenComponent
        flattenIntegrand = flattenComponentWeighted * flattenComponentWeighted
        numberOfGaussPoints = 3  # assuming some data applied around edges
        flattenGroupObjective = fieldmodule.createFieldMeshIntegral(
            flattenIntegrand, self._fitter.getModelReferenceCoordinatesField(), flattenMeshGroup)
        flattenGroupObjective.setNumbersOfPoints(numberOfGaussPoints)
        return flattenGroupObjective
