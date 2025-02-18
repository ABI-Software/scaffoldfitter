"""
Fit step for configuring subsequent behaviour, e.g. data projection settings.
"""

from scaffoldfitter.fitterstep import FitterStep
import logging
import sys


logger = logging.getLogger(__name__)


class FitterStepConfig(FitterStep):

    _jsonTypeId = "_FitterStepConfig"
    _centralProjectionToken = "centralProjection"
    _dataProportionToken = "dataProportion"
    _outlierLengthToken = "outlierLength"
    _projectionSubgroupToken = "projectionSubgroup"

    def __init__(self):
        super(FitterStepConfig, self).__init__()
        self._projectionCentreGroups = False

    @classmethod
    def getJsonTypeId(cls):
        return cls._jsonTypeId

    def decodeSettingsJSONDict(self, dctIn: dict):
        """
        Decode definition of step from JSON dict.
        """
        super().decodeSettingsJSONDict(dctIn)  # to decode group settings

    def encodeSettingsJSONDict(self) -> dict:
        """
        Encode definition of step in dict.
        :return: Settings in a dict ready for passing to json.dump.
        """
        # only has group settings for now
        return super().encodeSettingsJSONDict()

    def clearGroupCentralProjection(self, groupName):
        """
        Clear local group central projection so fall back to last config or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._centralProjectionToken)

    def getGroupCentralProjection(self, groupName):
        """
        Get flag controlling whether projections for this group are made with
        data points and target elements translated to a common central point.
        This can help fit groups which start well away from their targets.
        If not set or inherited, gets value from default group.
        :param groupName:  Exact model group name, or None for default group.
        :return:  Central projection flag, setLocally, inheritable.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        return self.getGroupSetting(groupName, self._centralProjectionToken, False)

    def setGroupCentralProjection(self, groupName, centralProjection):
        """
        Set flag controlling whether projections for this group are made with
        data points and target elements translated to a common central point.
        This can help fit groups which start well away from their targets.
        :param groupName:  Exact model group name, or None for default group.
        :param centralProjection:  Boolean True/False or None to reset to global
        default (False). Function ensures value is valid.
        """
        if centralProjection is not None:
            if not isinstance(centralProjection, bool):
                centralProjection = False
        self.setGroupSetting(groupName, self._centralProjectionToken, centralProjection)

    def clearGroupDataProportion(self, groupName):
        """
        Clear local group data proportion so fall back to last config or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._dataProportionToken)

    def getGroupDataProportion(self, groupName):
        """
        Get proportion of group data points to include in fit, from 0.0 (0%) to
        1.0 (100%), plus flags indicating where it has been set.
        If not set or inherited, gets value from default group.
        :param groupName:  Exact model group name, or None for default group.
        :return:  Proportion, setLocally, inheritable.
        Proportion of points used for group from 0.0 to 1.0 (default).
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        return self.getGroupSetting(groupName, self._dataProportionToken, 1.0)

    def setGroupDataProportion(self, groupName, proportion):
        """
        Set proportion of group data points to include in fit, or reset to
        global default.
        :param groupName:  Exact model group name, or None for default group.
        :param proportion:  Float valued proportion from 0.0 (0%) to 1.0 (100%),
        or None to reset to global default (1.0). Function ensures value is valid.
        """
        if proportion is not None:
            if not isinstance(proportion, float):
                proportion = self.getGroupDataProportion(groupName)[0]
            elif proportion < 0.0:
                proportion = 0.0
            elif proportion > 1.0:
                proportion = 1.0
        self.setGroupSetting(groupName, self._dataProportionToken, proportion)

    def clearGroupOutlierLength(self, groupName):
        """
        Clear local group outlier length so fall back to last config or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._outlierLengthToken)

    def getGroupOutlierLength(self, groupName):
        """
        Get relative or absolute length of data projections above which data points are treated as
        outliers and not included in the fit, plus flags indicating where it has been set.
        Values from -1.0 up to < 0.0 are negative proportions of maximum data projection to exclude,
        e.g. -0.1 excludes data points within 10% of the maximum data projection.
        Value 0.0 disables outlier filtering and includes all data (subject to other filters).
        Values > 0.0 gives absolute projection length above which data points are excluded.
        If not set or inherited, gets value from default group.
        :param groupName:  Exact model group name, or None for default group.
        :return:  OutlierLength, setLocally, inheritable.
        Absolute outlier length > 0.0, or 0.0 to include all data (default).
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        return self.getGroupSetting(groupName, self._outlierLengthToken, 0.0)

    def setGroupOutlierLength(self, groupName, outlierLength):
        """
        Set relative or absolute length of data projections above which data points are treated as
        outliers and not included in the fit.
        :param groupName:  Exact model group name, or None for default group.
        :param outlierLength:  From -1.0 to < 0.0: negative proportion to exclude, 0.0: disable
        outlier filter, > 0.0: absolute projection length above which data points are exclude,
        None to reset to global default (0.0).
        Function ensures value is valid.
        """
        if outlierLength is not None:
            if not isinstance(outlierLength, float):
                outlierLength = self.getGroupOutlierLength(groupName)[0]
            elif outlierLength < -1.0:
                outlierLength = -1.0
        self.setGroupSetting(groupName, self._outlierLengthToken, outlierLength)

    def clearGroupProjectionSubgroup(self, groupName):
        """
        Clear projection subgroup so fall back to last config or global default.
        :param groupName:  Exact model group name, or None for default group.
        """
        self.clearGroupSetting(groupName, self._projectionSubgroupToken)

    def getGroupProjectionSubgroup(self, groupName):
        """
        Get subgroup to intersect with the group's domain to restrict which elements
        are projected onto, or None.
        :param groupName:  Exact model group name, or None for default group.
        :return:  projectionSubgroup, setLocally, inheritable.
        Zinc FieldGroup if set otherwise None.
        The second return value is True if the value is set locally to a value
        or None if reset locally.
        The third return value is True if a previous config has set the value.
        """
        projectionSubgroupName, setLocally, inheritable = self.getGroupSetting(
            groupName, self._projectionSubgroupToken, None)
        if projectionSubgroupName:
            projectionSubgroup = self._fitter.getFieldmodule().findFieldByName(projectionSubgroupName).castGroup()
            if not projectionSubgroup.isValid():
                logger.error("Projection subgroup field not found. Has it been renamed?")
                projectionSubgroup = None
        else:
            projectionSubgroup = None
        return projectionSubgroup, setLocally, inheritable

    def setGroupProjectionSubgroup(self, groupName, projectionSubgroup):
        """
        Set subgroup to intersect with the group's domain to restrict which elements
        are projected onto, e.g. a lower dimensional subset.
        Stored internally by name, so will become invalid with renaming.
        :param groupName:  Exact model group name, or None for default group.
        :param projectionSubgroup:  A valid Zinc FieldGroup for this region, or None to reset to default None.
        """
        projectionSubgroupName = None
        if projectionSubgroup is not None:
            if not ((self._fitter.getRegion() == projectionSubgroup.getFieldmodule().getRegion()) and
                    projectionSubgroup.castGroup().isValid()):
                logger.error("Invalid projection subgroup.")
                return
            projectionSubgroupName = projectionSubgroup.getName()
        self.setGroupSetting(groupName, self._projectionSubgroupToken, projectionSubgroupName)

    def run(self, modelFileNameStem=None):
        """
        Calculate data projections with current settings.
        :param modelFileNameStem: Optional name stem of intermediate output file to write.
        """
        self._fitter.calculateDataProjections(self)
        if modelFileNameStem:
            self._fitter.writeModel(modelFileNameStem + "_config.exf")
        self.setHasRun(True)
