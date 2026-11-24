from cmlibs.maths.vectorops import cross, div, magnitude, sub
from cmlibs.zinc.result import RESULT_OK
import logging


logger = logging.getLogger(__name__)


class Mesh_location_group_boundary_axes:
    """
    Class for finding whether a mesh location is on the boundary of a group (from the highest dimension non-empty
    mesh group) and line/surface axes.
    """

    def __init__(self, mesh_group, coordinates, fieldcache):
        """
        Class for converting locations on a group into:
        1. True/False whether location is on boundary.
        2. line/surface axes
        Client should use this with ChangeManager(fieldmodule).
        :param mesh_group: The mesh group to work with. Currently limited to dimension 1-2.
        :param coordinates: Coordinate field on the mesh/group.
        :param fieldcache: Fieldcache for evaluating field with.
        """
        self._mesh_group = mesh_group
        self._mesh_dimension = mesh_group.getDimension()
        assert 1 <= self._mesh_dimension <= 2
        self._group = mesh_group.getFieldGroup()
        self._group_name = self._group.getName()
        self._fieldmodule = self._group.getFieldmodule()
        self._coordinates = coordinates
        self._coordinates_count = coordinates.getNumberOfComponents()
        self._fieldcache = fieldcache
        self._d1 = None
        self._d2 = None
        if self._mesh_dimension >= 1:
            self._d1 = self._mesh_group.getChartDifferentialoperator(1, 1)
            if self._mesh_dimension >= 2:
                self._d2 = self._mesh_group.getChartDifferentialoperator(1, 2)

    LOWER_XI_BOUNDARY_TOL = 1.0E-6  # how close to boundary of element's xi coordinates to be considered on the boundary
    UPPER_XI_BOUNDARY_TOL = 1.0 - LOWER_XI_BOUNDARY_TOL

    def get_line_axes(self, element, xi, element_id_to_node_ids: dict, node_id_to_element_ids: dict):
        """
        Determine whether the supplied mesh location (element, xi) is on the boundary of the 1-D group and the set of
        orthonormal axes at that point: tangent, normal1, normal2.
        :param element: Element the location is in.
        :param xi: Element 'xi' chart coordinates on [0.0, 1.0] as can only be a line.
        :param element_id_to_node_ids: Map from element id to [node ids in basis order] for elements in group.
        :param node_id_to_element_ids: Map from node id to list(element ids) it is in, for group.
        If in only one element then location is on the boundary.
        :return: On boundary (True if datapoint projects onto boundary of group, False if into interior),
        line axes: 3 coordinates: tangent, side1, side2
                   2 coordinates: tangent, side1
                   1 coordinate: tangent
        """
        assert self._mesh_dimension == 1
        element_id = element.getIdentifier()
        self._fieldcache.setMeshLocation(element, xi)
        result, xd1 = self._coordinates.evaluateDerivative(self._d1, self._fieldcache, self._coordinates_count)
        if result != RESULT_OK:
            logger.error("get_line_axes.  Coordinates not defined on element " + str(element_id) +
                         " of group " + self._group_name)
            return False, [0.0] * (self._coordinates_count * self._coordinates_count)
        if self._coordinates_count == 1:
            xd1 = [xd1]  # a quirk of swigzinc: scalars are returned as float

        mag_xd1 = magnitude(xd1)
        if mag_xd1 <= 0.0:
            tangent = [0.0] * self._coordinates_count
            side1 = ([0.0] * self._coordinates_count) if (self._coordinates_count > 1) else []
            side2 = ([0.0] * self._coordinates_count) if (self._coordinates_count > 2) else []
        else:
            tangent = div(xd1, mag_xd1)
            if self._coordinates_count == 3:
                side1 = cross(tangent, [1.0, 0.0, 0.0])
                mag_side1 = magnitude(side1)
                if mag_side1 < 0.1:
                    side1 = cross(tangent, [0.0, 1.0, 0.0])
                    mag_side1 = magnitude(side1)
                if mag_side1 <= 0.0:
                    side2 = [0.0, 0.0, 0.0]
                else:
                    side1 = div(side1, mag_side1)
                    side2 = cross(tangent, side1)
            elif self._coordinates_count == 2:
                side1 = [-tangent[1], tangent[0]]
                side2 = []
            else:
                side1 = side2 = []

        on_boundary = False
        node_id = (
            element_id_to_node_ids[element_id][0] if (xi < self.LOWER_XI_BOUNDARY_TOL) else
            element_id_to_node_ids[element_id][-1] if (xi > self.UPPER_XI_BOUNDARY_TOL) else None)
        if node_id is not None:
            on_boundary = len(node_id_to_element_ids[node_id]) == 1
        line_axes = tangent + side1 + side2
        return on_boundary, line_axes

    def _is_on_group_boundary(self, face_element):
        """
        Determine whether a line is on the surface group boundary. Assumes one parent element is in group.
        :param face_element: Line face of surface element in group.
        :return: True if face_element has exactly one parent element in mesh group, otherwise False.
        """
        parents_count = face_element.getNumberOfParents()
        if parents_count == 1:
            return True
        parents_count_in_group = 0
        for p in range(1, parents_count + 1):
            parent = face_element.getParentElement(p)
            if self._mesh_group.containsElement(parent):
                parents_count_in_group += 1
                if parents_count_in_group > 1:
                    return False
        return True

    def get_surface_axes(self, element, xi):
        """
        Determine whether the supplied mesh location (element, xi) is on the boundary of the 2-D group and the set of
        orthonormal surface axes at that point: tangent1, tangent2[, normal].
        :param element: Element the location is in.
        :param xi: Element 'xi' chart coordinates on [0.0, 1.0] as can only be a line.
        :return: On boundary (True if on edge of group, False if in interior),
        surface axes: 3 coordinates: tangent1 (normal to boundary edge),
                                     tangent2 (normal to tangent1, along edge if on boundary), surface normal;
                      2 coordinates: tangent1, (normal to boundary edge), tangent2
        """
        assert self._mesh_dimension == 2
        assert self._coordinates_count >= 2
        element_id = element.getIdentifier()
        self._fieldcache.setMeshLocation(element, xi)
        result1, xd1 = self._coordinates.evaluateDerivative(self._d1, self._fieldcache, self._coordinates_count)
        result2, xd2 = self._coordinates.evaluateDerivative(self._d2, self._fieldcache, self._coordinates_count)
        if (result1 != RESULT_OK) or (result2 != RESULT_OK):
            logger.error("get_surface_axes.  Coordinates not defined on element " + str(element_id) +
                         " of group " + self._group_name)
            return False, [0.0] * (self._coordinates_count * self._coordinates_count)

        mag_xd1 = magnitude(xd1)
        mag_xd2 = magnitude(xd2)
        if (mag_xd1 <= 0.0) and (mag_xd2 <= 0.0):
            logger.error("get_surface_axes.  Coordinates d/xi1 and d/xi2 both zero at element " + str(element_id) +
                         " xi " + str(xi) + " for group " + self._group_name)
            return False, [0.0] * (self._coordinates_count * self._coordinates_count)
        tangent1 = div(xd1, mag_xd1) if (mag_xd1 > 0.0) else xd1
        tangent2 = div(xd2, mag_xd2) if (mag_xd2 > 0.0) else xd2
        if self._coordinates_count == 3:
            normal = cross(xd1, xd2)
            mag_normal = magnitude(normal)
            normal = div(normal, mag_normal) if (mag_normal > 0.0) else normal
        else:
            normal = []

        on_boundary = False
        on_boundaries = [False] * 4
        if xi[0] < self.LOWER_XI_BOUNDARY_TOL:
            on_boundaries[0] = self._is_on_group_boundary(element.getFaceElement(1))
        shape_type = element.getShapeType()
        if shape_type == element.SHAPE_TYPE_SQUARE:
            if xi[0] > self.UPPER_XI_BOUNDARY_TOL:
                on_boundaries[1] = self._is_on_group_boundary(element.getFaceElement(2))
            if xi[1] < self.LOWER_XI_BOUNDARY_TOL:
                on_boundaries[2] = self._is_on_group_boundary(element.getFaceElement(3))
            elif xi[1] > self.UPPER_XI_BOUNDARY_TOL:
                on_boundaries[3] = self._is_on_group_boundary(element.getFaceElement(4))
            if any(on_boundaries):
                on_boundary = True
                if on_boundaries[2] or on_boundaries[3]:
                    tangent2 = tangent1
        elif shape_type == element.SHAPE_TYPE_TRIANGLE:
            if xi[1] < self.LOWER_XI_BOUNDARY_TOL:
                on_boundaries[1] = self._is_on_group_boundary(element.getFaceElement(2))
            nu = xi[0] + xi[1]
            if nu > self.UPPER_XI_BOUNDARY_TOL:
                on_boundaries[2] = self._is_on_group_boundary(element.getFaceElement(3))
            if any(on_boundaries):
                on_boundary = True
                if on_boundaries[1]:
                    tangent2 = tangent1
                elif on_boundaries[2]:
                    tangent2 = sub(xd1, xd2)
                    mag_tangent2 = magnitude(tangent2)
                    tangent2 = div(tangent2, mag_tangent2) if (mag_tangent2 > 0.0) else tangent2
        # make tangent1 orthogonal to tangent2
        if self._coordinates_count == 3:
            tangent1 = cross(tangent2, normal)
        else:
            tangent1 = [-tangent2[1], tangent2[0]]
        surface_axes = tangent1 + tangent2 + normal

        return on_boundary, surface_axes
