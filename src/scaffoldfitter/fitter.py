"""
Main class for fitting scaffolds.
"""

import json
from opencmiss.zinc.context import Context
from opencmiss.zinc.field import Field, FieldFindMeshLocation, FieldGroup
from opencmiss.zinc.result import RESULT_OK, RESULT_WARNING_PART_DONE
from scaffoldfitter.utils.zinc_utils import assignFieldParameters, createFieldClone, evaluateNodesetMeanCoordinates, \
    findNodeWithName, getGroupList, getOrCreateFieldFiniteElement, getOrCreateFieldMeshLocation, getUniqueFieldName, ZincCacheChanges


class Fitter:

    def __init__(self, zincModelFileName, zincDataFileName):
        self._context = Context("Scaffoldfitter")
        self._region = None
        self._fieldmodule = None
        self._zincModelFileName = zincModelFileName
        self._zincDataFileName = zincDataFileName
        self._modelCoordinatesField = None
        self._modelCoordinatesFieldName = None
        self._modelReferenceCoordinatesField = None
        self._dataCoordinatesField = None
        self._dataCoordinatesFieldName = None
        self._mesh = []  # [dimension - 1]
        self._dataProjectionMeshLocationField = [ ]  # [dimension - 1]
        self._dataProjectionNodeGroupField = []  # [dimension - 1]
        self._dataProjectionNodesetGroup = []  # [dimension - 1]
        self._dataProjectionDirectionField = None  # for storing original projection direction unit vector
        self._markerGroup = None
        self._markerGroupName = None
        self._markerNodeGroup = None
        self._markerLocationField = None
        self._markerNameField = None
        self._markerCoordinatesField = None
        self._markerDataGroup = None
        self._markerDataCoordinatesField = None
        self._markerDataNameField = None
        self._markerDataLocationField = None
        self._markerDataLocationCoordinatesField = None
        self._markerDataLocationGroupField = None
        self._markerDataLocationGroup = None
        self._diagnosticLevel = 0
        self._fitterSteps = []
        self.loadModel()

    def loadModel(self):
        self._region = self._context.createRegion()
        self._fieldmodule = self._region.getFieldmodule()
        result = self._region.readFile(self._zincModelFileName)
        assert result == RESULT_OK, "Failed to load model file" + str(self._zincModelFileName)
        result = self._region.readFile(self._zincDataFileName)
        assert result == RESULT_OK, "Failed to load data file" + str(self._zincDataFileName)
        self._mesh = [ self._fieldmodule.findMeshByDimension(d + 1) for d in range(3) ]
        self._discoverModelCoordinatesField()
        self._discoverDataCoordinatesField()
        self._discoverMarkerGroup()
        self._defineDataProjectionFields()
        self.calculateDataProjections()

    def getDataCoordinatesField(self):
        return self._dataCoordinatesField

    def setDataCoordinatesField(self, dataCoordinatesField : Field):
        finiteElementField = dataCoordinatesField.castFiniteElement()
        assert finiteElementField.isValid() and (finiteElementField.getNumberOfComponents() == 3)
        self._dataCoordinatesFieldName = dataCoordinatesField.getName()
        self._dataCoordinatesField = finiteElementField

    def setDataCoordinatesFieldByName(self, dataCoordinatesFieldName):
        self.setDataCoordinatesField(self._fieldmodule.findFieldByName(dataCoordinatesFieldName))

    def _discoverDataCoordinatesField(self):
        """
        Choose default dataCoordinates field.
        """
        self._dataCoordinatesField = None
        field = None
        if self._dataCoordinatesFieldName:
            field = self._fieldmodule.findFieldByName(self._dataCoordinatesFieldName)
        else:
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            datapoint = datapoints.createNodeiterator().next()
            if datapoint.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setNode(datapoint)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isTypeCoordinate() and (field.getNumberOfComponents() == 3) and (field.castFiniteElement().isValid()):
                        if field.isDefinedAtLocation(fieldcache):
                            break;
                    field = fielditer.next()
                else:
                    field = None
        self.setDataCoordinatesField(field)

    def getMarkerGroup(self):
        return self._markerGroup

    def setMarkerGroup(self, markerGroup : Field):
        self._markerGroup = None
        self._markerGroupName = None
        self._markerNodeGroup = None
        self._markerLocationField = None
        self._markerCoordinatesField = None
        self._markerNameField = None
        self._markerDataGroup = None
        self._markerDataCoordinatesField = None
        self._markerDataNameField = None
        self._markerDataLocationField = None
        self._markerDataLocationCoordinatesField = None
        self._markerDataLocationGroupField = None
        self._markerDataLocationGroup = None
        if not markerGroup:
            return
        fieldGroup = markerGroup.castGroup()
        assert fieldGroup.isValid()
        self._markerGroup = fieldGroup
        self._markerGroupName = markerGroup.getName()
        nodes = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        self._markerNodeGroup = self._markerGroup.getFieldNodeGroup(nodes).getNodesetGroup()
        if self._markerNodeGroup.isValid():
            node = self._markerNodeGroup.createNodeiterator().next()
            if node.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setNode(node)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isDefinedAtLocation(fieldcache):
                        if (not self._markerLocationField) and field.castStoredMeshLocation().isValid():
                            self._markerLocationField = field
                        elif (not self._markerNameField) and (field.getValueType() == Field.VALUE_TYPE_STRING):
                            self._markerNameField = field
                    field = fielditer.next()
                self._updateMarkerCoordinatesField()
        else:
            self._markerNodeGroup = None
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        self._markerDataGroup = self._markerGroup.getFieldNodeGroup(datapoints).getNodesetGroup()
        if self._markerDataGroup.isValid():
            datapoint = self._markerDataGroup.createNodeiterator().next()
            if datapoint.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setNode(datapoint)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isDefinedAtLocation(fieldcache):
                        if (not self._markerDataCoordinatesField) and field.isTypeCoordinate() and \
                                (field.getNumberOfComponents() == 3) and (field.castFiniteElement().isValid()):
                            self._markerDataCoordinatesField = field
                        elif (not self._markerDataNameField) and (field.getValueType() == Field.VALUE_TYPE_STRING):
                            self._markerDataNameField = field
                    field = fielditer.next()
        else:
            self._markerDataGroup = None
        self._calculateMarkerDataLocations()

    def setMarkerGroupByName(self, markerGroupName):
        self.setMarkerGroup(self._fieldmodule.findFieldByName(markerGroupName))

    def getMarkerDataFields(self):
        """
        Only call if markerGroup exists.
        :return: markerDataGroup, markerDataCoordinates, markerDataName
        """
        return self._markerDataGroup, self._markerDataCoordinatesField, self._markerDataNameField

    def getMarkerModelFields(self):
        """
        Only call if markerGroup exists.
        :return: markerNodeGroup, markerLocation, markerCoordinates, markerName
        """
        return self._markerNodeGroup, self._markerLocationField, self._markerCoordinatesField, self._markerNameField

    def _calculateMarkerDataLocations(self):
        """
        Called when markerGroup exists.
        Find matching marker mesh locations for marker data points.
        Only finds matching location where there is one datapoint and one node
        for each name in marker group.
        Defines datapoint group self._markerDataLocationGroup to contain those with locations.
        """
        self._markerDataLocationField = None
        self._markerDataLocationCoordinatesField = None
        self._markerDataLocationGroupField = None
        self._markerDataLocationGroup = None
        if not (self._markerDataGroup and self._markerDataNameField and self._markerNodeGroup and self._markerLocationField and self._markerNameField):
            return

        markerPrefix = self._markerGroup.getName()
        # assume marker locations are in highest dimension mesh
        mesh = self.getHighestDimensionMesh()
        datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
        meshDimension = mesh.getDimension()
        fieldcache = self._fieldmodule.createFieldcache()
        with ZincCacheChanges(self._fieldmodule):
            self._markerDataLocationField = getOrCreateFieldMeshLocation(self._fieldmodule, mesh, markerPrefix + "_data_location_")
            self._markerDataLocationGroupField = self._fieldmodule.createFieldNodeGroup(datapoints)
            self._markerDataLocationGroupField.setName(getUniqueFieldName(self._fieldmodule, markerPrefix + "_data_location_group"))
            self._markerDataLocationGroup = self._markerDataLocationGroupField.getNodesetGroup()
            self._updateMarkerDataLocationCoordinatesField()
            nodetemplate = self._markerDataGroup.createNodetemplate()
            nodetemplate.defineField(self._markerDataLocationField)
            datapointIter = self._markerDataGroup.createNodeiterator()
            datapoint = datapointIter.next()
            while datapoint.isValid():
                fieldcache.setNode(datapoint)
                name = self._markerDataNameField.evaluateString(fieldcache)
                # if this is the only datapoint with name:
                if name and findNodeWithName(self._markerDataGroup, self._markerDataNameField, name):
                    node = findNodeWithName(self._markerNodeGroup, self._markerNameField, name)
                    if node:
                        fieldcache.setNode(node)
                        element, xi = self._markerLocationField.evaluateMeshLocation(fieldcache, meshDimension)
                        if element.isValid():
                            datapoint.merge(nodetemplate)
                            fieldcache.setNode(datapoint)
                            self._markerDataLocationField.assignMeshLocation(fieldcache, element, xi)
                            self._markerDataLocationGroup.addNode(datapoint)
                datapoint = datapointIter.next()
        # Warn about datapoints without a location in model
        markerDataGroupSize = self._markerDataGroup.getSize()
        markerDataLocationGroupSize = self._markerDataLocationGroup.getSize()
        markerNodeGroupSize = self._markerNodeGroup.getSize()
        if self.getDiagnosticLevel() > 0:
            if markerDataLocationGroupSize < markerDataGroupSize:
                print("Warning: Only " + str(markerDataLocationGroupSize) + " of " + str(markerDataGroupSize) + " marker data points have model locations")
            if markerDataLocationGroupSize < markerNodeGroupSize:
                print("Warning: Only " + str(markerDataLocationGroupSize) + " of " + str(markerNodeGroupSize) + " marker model locations used")

    def _discoverMarkerGroup(self):
        self._markerGroup = None
        self._markerNodeGroup = None
        self._markerLocationField = None
        self._markerNameField = None
        self._markerCoordinatesField = None
        markerGroup = self._fieldmodule.findFieldByName(self._markerGroupName if self._markerGroupName else "marker")
        if not markerGroup.castGroup().isValid():
            markerGroup = None
        self.setMarkerGroup(markerGroup)

    def _updateMarkerCoordinatesField(self):
        if self._modelCoordinatesField and self._markerLocationField:
            with ZincCacheChanges(self._fieldmodule):
                self._markerCoordinatesField = self._fieldmodule.createFieldEmbedded(self._modelCoordinatesField, self._markerLocationField)
                markerPrefix = self._markerGroup.getName()
                self._markerCoordinatesField.setName(getUniqueFieldName(self._fieldmodule, markerPrefix + "_coordinates"))
        else:
            self._markerCoordinatesField = None

    def _updateMarkerDataLocationCoordinatesField(self):
        if self._modelCoordinatesField and self._markerDataLocationField:
            with ZincCacheChanges(self._fieldmodule):
                self._markerDataLocationCoordinatesField = self._fieldmodule.createFieldEmbedded(self._modelCoordinatesField, self._markerDataLocationField)
                markerPrefix = self._markerGroup.getName()
                self._markerDataLocationCoordinatesField.setName(getUniqueFieldName(self._fieldmodule, markerPrefix + "_data_location_coordinates"))
        else:
            self._markerDataLocationCoordinatesField = None

    def getModelCoordinatesField(self):
        return self._modelCoordinatesField

    def getModelReferenceCoordinatesField(self):
        return self._modelReferenceCoordinatesField

    def setModelCoordinatesField(self, modelCoordinatesField : Field):
        finiteElementField = modelCoordinatesField.castFiniteElement()
        assert finiteElementField.isValid() and (finiteElementField.getNumberOfComponents() == 3)
        self._modelCoordinatesField = finiteElementField
        self._modelCoordinatesFieldName = modelCoordinatesField.getName()
        self._modelReferenceCoordinatesField = createFieldClone(self._modelCoordinatesField, "reference_" + self._modelCoordinatesField.getName())
        self._updateMarkerCoordinatesField()
        self._updateMarkerDataLocationCoordinatesField()

    def setModelCoordinatesFieldByName(self, modelCoordinatesFieldName):
        self.setModelCoordinatesField(self._fieldmodule.findFieldByName(modelCoordinatesFieldName))

    def _discoverModelCoordinatesField(self):
        """
        Choose default modelCoordinates field.
        """
        self._modelCoordinatesField = None
        self._modelReferenceCoordinatesField = None
        field = None
        if self._modelCoordinatesFieldName:
            field = self._fieldmodule.findFieldByName(self._modelCoordinatesFieldName)
        else:
            mesh = self.getHighestDimensionMesh()
            element = mesh.createElementiterator().next()
            if element.isValid():
                fieldcache = self._fieldmodule.createFieldcache()
                fieldcache.setElement(element)
                fielditer = self._fieldmodule.createFielditerator()
                field = fielditer.next()
                while field.isValid():
                    if field.isTypeCoordinate() and (field.getNumberOfComponents() == 3) and (field.castFiniteElement().isValid()):
                        if field.isDefinedAtLocation(fieldcache):
                            break;
                    field = fielditer.next()
                else:
                    field = None
        if field:
            self.setModelCoordinatesField(field)

    def runConfig(self):
        """
        Complete configuration setup.
        """
        self._hasRunConfig = True

    def _defineDataProjectionFields(self):
        self._dataProjectionMeshLocationField = []
        self._dataProjectionNodeGroupField = []
        self._dataProjectionNodesetGroup = []
        with ZincCacheChanges(self._fieldmodule):
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            for d in range(2):
                mesh = self._mesh[d]
                self._dataProjectionMeshLocationField.append(getOrCreateFieldMeshLocation(self._fieldmodule, mesh, namePrefix = "data_projection_location_"))
                field = self._fieldmodule.createFieldNodeGroup(datapoints)
                field.setName(getUniqueFieldName(self._fieldmodule, "data_projection_group_" + mesh.getName()))
                self._dataProjectionNodeGroupField.append(field)
                self._dataProjectionNodesetGroup.append(field.getNodesetGroup())
            self._dataProjectionDirectionField = getOrCreateFieldFiniteElement(self._fieldmodule, "data_projection_direction",
                componentsCount = 3, componentNames = [ "x", "y", "z" ])

    def calculateDataProjections(self):
        """
        Find projections of datapoints' coordinates onto model coordinates,
        by groups i.e. from datapoints group onto matching 2-D or 1-D mesh group.
        Calculate and store projection direction unit vector.
        """
        assert self._dataCoordinatesField and self._modelCoordinatesField
        with ZincCacheChanges(self._fieldmodule):
            findMeshLocation = None
            datapoints = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS)
            fieldcache = self._fieldmodule.createFieldcache()
            for d in range(2):
                self._dataProjectionNodesetGroup[d].removeAllNodes()
            groups = getGroupList(self._fieldmodule)
            for group in groups:
                groupName = group.getName()
                dataGroup = group.getFieldNodeGroup(datapoints).getNodesetGroup()
                if not dataGroup.isValid():
                    continue
                for dimension in range(2, 0, -1):
                    meshGroup = group.getFieldElementGroup(self._mesh[dimension - 1]).getMeshGroup()
                    if meshGroup.isValid() and (meshGroup.getSize() > 0):
                        break
                else:
                    if self.getDiagnosticLevel() > 0:
                        print("Fit Geometry:  Warning: Cannot project data for group " + groupName + " as no matching mesh group")
                    continue
                meshLocation = self._dataProjectionMeshLocationField[dimension - 1]
                dataProjectionNodesetGroup = self._dataProjectionNodesetGroup[dimension - 1]
                nodeIter = dataGroup.createNodeiterator()
                node = nodeIter.next()
                fieldcache.setNode(node)
                if not self._dataCoordinatesField.isDefinedAtLocation(fieldcache):
                    if self.getDiagnosticLevel() > 0:
                        print("Fit Geometry:  Warning: Cannot project data for group " + groupName + " as field " + self._dataCoordinatesField.getName() + " is not defined on data")
                    continue
                if not meshLocation.isDefinedAtLocation(fieldcache):
                    # define meshLocation and on data Group:
                    nodetemplate = datapoints.createNodetemplate()
                    nodetemplate.defineField(meshLocation)
                    nodetemplate.defineField(self._dataProjectionDirectionField)
                    while node.isValid():
                        result = node.merge(nodetemplate)
                        #print("node",node.getIdentifier(),"result",result)
                        node = nodeIter.next()
                    del nodetemplate
                    # restart iteration
                    nodeIter = dataGroup.createNodeiterator()
                    node = nodeIter.next()
                findMeshLocation = self._fieldmodule.createFieldFindMeshLocation(self._dataCoordinatesField, self._modelCoordinatesField, meshGroup)
                findMeshLocation.setSearchMode(FieldFindMeshLocation.SEARCH_MODE_NEAREST)
                while node.isValid():
                    fieldcache.setNode(node)
                    element, xi = findMeshLocation.evaluateMeshLocation(fieldcache, dimension)
                    if not element.isValid():
                        print("Fit Geometry:  Error finding data projection nearest mesh location for group " + groupName + ". Aborting group.")
                        break
                    result = meshLocation.assignMeshLocation(fieldcache, element, xi)
                    #print(result, "node", node.getIdentifier(), "element", element.getIdentifier(), "xi", xi)
                    #if result != RESULT_OK:
                    #    mesh = meshLocation.getMesh()
                    #    print("--> mesh", mesh.isValid(), mesh.getDimension(), findMeshLocation.getMesh().getDimension())
                    #    print("node", node.getIdentifier(), "is defined", meshLocation.isDefinedAtLocation(fieldcache))
                    assert result == RESULT_OK, "Fit Geometry:  Failed to assign data projection mesh location for group " + groupName
                    dataProjectionNodesetGroup.addNode(node)
                    node = nodeIter.next()

            # Store data projection directions
            for dimension in range(1, 3):
                nodesetGroup = self._dataProjectionNodesetGroup[dimension - 1]
                if nodesetGroup.getSize() > 0:
                    fieldassignment = self._dataProjectionDirectionField.createFieldassignment(
                        self._fieldmodule.createFieldNormalise(self._fieldmodule.createFieldSubtract(
                            self._fieldmodule.createFieldEmbedded(self._modelCoordinatesField, self._dataProjectionMeshLocationField[dimension - 1]),
                            self._dataCoordinatesField)))
                    fieldassignment.setNodeset(nodesetGroup)
                    result = fieldassignment.assign()
                    assert result in [ RESULT_OK, RESULT_WARNING_PART_DONE ], \
                        "Fit Geometry:  Failed to assign data projection directions for dimension " + str(dimension)

            if self.getDiagnosticLevel() > 0:
                # Warn about unprojected points
                unprojectedDatapoints = self._fieldmodule.createFieldNodeGroup(datapoints).getNodesetGroup()
                unprojectedDatapoints.addNodesConditional(self._fieldmodule.createFieldIsDefined(self._dataCoordinatesField))
                for d in range(2):
                    unprojectedDatapoints.removeNodesConditional(self._dataProjectionNodeGroupField[d])
                unprojectedCount = unprojectedDatapoints.getSize()
                if unprojectedCount > 0:
                    print("Warning: " + str(unprojected) + " data points with data coordinates have not been projected")
                del unprojectedDatapoints

            # remove temporary objects before clean up ZincCacheChanges
            del findMeshLocation
            del fieldcache

    def _addFitterStep(self, fitterStep):
        self._fitterSteps.append(fitterStep)

    def _removeFitterStep(self, fitterStep):
        self._fitterSteps.remove(fitterStep)

    def getNextFitterStep(self, refFitterStep):
        """
        Return next fitter step after refFitterStep, or before if last, otherwise None.
        """
        index = self._fitterSteps.index(refFitterStep) + 1
        if index >= len(self._fitterSteps):
            index -= 2
        if index < 0:
            return None
        return self._fitterSteps[index]

    def getDataProjectionDirectionField(self):
        return self._dataProjectionDirectionField

    def getDataProjectionNodeGroupField(self, dimension):
        assert 1 <= dimension <= 2
        return self._dataProjectionNodeGroupField[dimension - 1]

    def getDataProjectionNodesetGroup(self, dimension):
        assert 1 <= dimension <= 2
        return self._dataProjectionNodesetGroup[dimension - 1]

    def getDataProjectionMeshLocationField(self, dimension):
        assert 1 <= dimension <= 2
        return self._dataProjectionMeshLocationField[dimension - 1]

    def getMarkerDataLocationNodesetGroup(self):
        return self._markerDataLocationGroup

    def getMarkerDataLocationField(self):
        return self._markerDataLocationField

    def getContext(self):
        return self._context

    def getRegion(self):
        return self._region

    def getFieldmodule(self):
        return self._fieldmodule

    def getFitterSteps(self):
        return self._fitterSteps

    def getMesh(self, dimension):
        assert 1 <= dimension <= 3
        return self._mesh[dimension - 1]

    def getHighestDimensionMesh(self):
        """
        :return: Highest dimension mesh with elements in it, or None if none.
        """
        for d in range(2, -1, -1):
            mesh = self._mesh[d]
            if mesh.getSize() > 0:
                return mesh
        return None

    def evaluateNodeGroupMeanCoordinates(self, groupName, coordinatesFieldName, isData = False):
        group = self._fieldmodule.findFieldByName(groupName).castGroup()
        assert group.isValid()
        nodeset = self._fieldmodule.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_DATAPOINTS if isData else Field.DOMAIN_TYPE_NODES)
        nodesetGroup = group.getFieldNodeGroup(nodeset).getNodesetGroup()
        assert nodesetGroup.isValid()
        coordinates = self._fieldmodule.findFieldByName(coordinatesFieldName)
        return evaluateNodesetMeanCoordinates(coordinates, nodesetGroup)

    def getDiagnosticLevel(self):
        return self._diagnosticLevel

    def setDiagnosticLevel(self, diagnosticLevel):
        """
        :param diagnosticLevel: 0 = no diagnostic messages. 1 = Information and warning messages. 2 = Also optimisation reports.
        """
        assert diagnosticLevel >= 0
        self._diagnosticLevel = diagnosticLevel

    def updateModelReferenceCoordinates(self):
        assignFieldParameters(self._modelReferenceCoordinatesField, self._modelCoordinatesField)

    def writeModel(self, fileName):
        sir = self._region.createStreaminformationRegion()
        sr = sir.createStreamresourceFile(fileName)
        sir.setFieldNames([ self._modelCoordinatesField.getName() ])
        sir.setResourceDomainTypes(sr, Field.DOMAIN_TYPE_NODES)
        self._region.write(sir)

    def writeData(self, fileName):
        sir = self._region.createStreaminformationRegion()
        sr = sir.createStreamresourceFile(fileName)
        sir.setResourceDomainTypes(sr, Field.DOMAIN_TYPE_DATAPOINTS)
        self._region.write(sir)


class FitterStep:
    """
    Base class for fitter steps.
    """

    def __init__(self, fitter : Fitter):
        """
        Construct and add to Fitter.
        """
        self._fitter = fitter
        fitter._addFitterStep(self)
        self._hasRun = False

    def destroy(self):
        """
        Remove from Fitter.
        """
        self._fitter._removeFitterStep(self)
        self._fitter = None

    def getFitter(self):
        return self._fitter

    def hasRun(self):
        return self._hasRun

    def setHasRun(self, hasRun):
        self._hasRun = hasRun

    def getDiagnosticLevel(self):
        return self._fitter.getDiagnosticLevel()

    def run(self):
        """
        Override to perform action of derived FitStep
        """
        pass