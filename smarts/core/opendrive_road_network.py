# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import heapq
import logging
import math
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Set, Tuple
import trimesh
import trimesh.scene
from trimesh.exchange import gltf
import numpy as np
from cached_property import cached_property
from lxml import etree
from opendrive2lanelet.opendriveparser.elements.geometry import Line as LineGeometry
from opendrive2lanelet.opendriveparser.elements.opendrive import (
    OpenDrive as OpenDriveElement,
)
from opendrive2lanelet.opendriveparser.elements.road import Road as RoadElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import Lane as LaneElement
from opendrive2lanelet.opendriveparser.elements.roadLanes import (
    LaneOffset as LaneOffsetElement,
)
from opendrive2lanelet.opendriveparser.elements.roadLanes import (
    LaneSection as LaneSectionElement,
)
from opendrive2lanelet.opendriveparser.elements.roadLanes import (
    LaneWidth as LaneWidthElement,
)
from opendrive2lanelet.opendriveparser.elements.roadPlanView import (
    PlanView as PlanViewElement,
)
from opendrive2lanelet.opendriveparser.parser import parse_opendrive
from shapely.geometry import Polygon
import rtree

from smarts.core.road_map import RoadMap
from smarts.core.utils.math import (
    CubicPolynomial,
    constrain_angle,
    distance_point_to_polygon,
    get_linear_segments_for_range,
    offset_along_shape,
    position_at_shape_offset,
)

from .lanepoints import LinkedLanePoint, LanePoints
from .coordinates import BoundingBox, Point, Pose, RefLinePoint
from smarts.core.utils.geometry import generate_mesh_from_polygons


def _convert_camera(camera):
    result = {
        "name": camera.name,
        "type": "perspective",
        "perspective": {
            "aspectRatio": camera.fov[0] / camera.fov[1],
            "yfov": np.radians(camera.fov[1]),
            "znear": float(camera.z_near),
            # HACK: The trimesh gltf export doesn't include a zfar which Panda3D GLB
            #       loader expects. Here we override to make loading possible.
            "zfar": float(camera.z_near + 100),
        },
    }
    return result


gltf._convert_camera = _convert_camera


class _GLBData:
    def __init__(self, bytes_):
        self._bytes = bytes_

    def write_glb(self, output_path):
        with open(output_path, "wb") as f:
            f.write(self._bytes)


@dataclass
class LaneBoundary:
    refline: PlanViewElement
    inner: "LaneBoundary"
    lane_widths: List[LaneWidthElement]
    lane_offsets: List[LaneOffsetElement]
    segment_size: float = 0.5

    def refline_to_linear_segments(self, s_start: float, s_end: float) -> List[float]:
        s_vals = []
        geom_start = 0
        for geom in self.refline._geometries:
            geom_end = geom_start + geom.length
            if type(geom) == LineGeometry:
                s_vals.extend([geom_start, geom_end])
            else:
                s_vals.extend(
                    get_linear_segments_for_range(
                        geom_start, geom_end, self.segment_size
                    )
                )
            geom_start = geom_start + geom.length
        return [s for s in s_vals if s_start <= s <= s_end]

    def get_lane_offset(self, s: float) -> float:
        if len(self.lane_offsets) == 0:
            return 0
        i = 0
        if s < self.lane_offsets[0].start_pos:
            return 0
        while i < len(self.lane_offsets) - 1:
            if (
                self.lane_offsets[i].start_pos
                <= s
                <= self.lane_offsets[i + 1].start_pos
            ):
                break
            i += 1

        poly = CubicPolynomial.from_list(self.lane_offsets[i].polynomial_coefficients)
        ds = s - self.lane_offsets[i].start_pos
        offset = poly.eval(ds)
        return offset

    def lane_width_at_offset(self, offset: float) -> LaneWidthElement:
        i = 0
        while i < len(self.lane_widths) - 1:
            if (
                self.lane_widths[i].start_offset
                <= offset
                <= self.lane_widths[i + 1].start_offset
            ):
                break
            i += 1

        return self.lane_widths[i]

    def calc_t(self, s: float, section_s_start, lane_id) -> float:
        # Find the lateral shift of lane reference line with road reference line (known as laneOffset in OpenDRIVE)
        lane_offset = self.get_lane_offset(s)

        if not self.inner:
            return np.sign(lane_id) * lane_offset

        width = self.lane_width_at_offset(s - section_s_start)
        poly = CubicPolynomial.from_list(width.polynomial_coefficients)

        return poly.eval(s - section_s_start - width.start_offset) + self.inner.calc_t(
            s, section_s_start, lane_id
        )

    def to_linear_segments(self, s_start: float, s_end: float):
        if self.inner:
            inner_s_vals = self.inner.to_linear_segments(s_start, s_end)
        else:
            if self.lane_offsets:
                return get_linear_segments_for_range(s_start, s_end, self.segment_size)
            return self.refline_to_linear_segments(s_start, s_end)

        outer_s_vals = []
        curr_s_start = s_start
        for width in self.lane_widths:
            poly = CubicPolynomial.from_list(width.polynomial_coefficients)
            if poly.c == 0 and poly.d == 0:
                # Special case - only 2 vertices required
                outer_s_vals.extend([curr_s_start, curr_s_start + width.length])
            else:
                outer_s_vals.extend(
                    get_linear_segments_for_range(
                        curr_s_start, curr_s_start + width.length, self.segment_size
                    )
                )
            curr_s_start += width.length

        return sorted(set(inner_s_vals + outer_s_vals))


class OpenDriveRoadNetwork(RoadMap):
    DEFAULT_LANE_WIDTH = 3.2

    def __init__(self, xodr_file: str, default_lane_width=None, lanepoint_spacing=None):
        self._log = logging.getLogger(self.__class__.__name__)
        self._log.setLevel(logging.INFO)
        self._xodr_file = xodr_file
        self._default_lane_width = (
            default_lane_width
            if default_lane_width is not None
            else OpenDriveRoadNetwork.DEFAULT_LANE_WIDTH
        )
        self._surfaces: Dict[str, OpenDriveRoadNetwork.Surface] = {}
        self._roads: Dict[str, OpenDriveRoadNetwork.Road] = {}
        self._lanes: Dict[str, OpenDriveRoadNetwork.Lane] = {}
        self._lanepoints = None

        # Reference to lanes' R tree
        self._lane_rtree = None
        # To preserve a specific order of lanes for building R tree
        self._all_lanes = []

        self.load()

        if lanepoint_spacing is not None:
            assert lanepoint_spacing > 0
            self._lanepoints = LanePoints.from_opendrive(
                self, spacing=lanepoint_spacing
            )

    @classmethod
    def from_file(
        cls,
        xodr_file,
        default_lane_width=None,
        lanepoint_spacing=None,
    ):
        od_map = cls(
            xodr_file,
            default_lane_width=default_lane_width,
            lanepoint_spacing=lanepoint_spacing,
        )
        return od_map

    @staticmethod
    def _elem_id(elem, suffix):
        if type(elem) == LaneSectionElement:
            return f"{elem.parentRoad.id}_{elem.idx}_{suffix}"
        else:
            assert type(elem) == LaneElement
            return f"{elem.parentRoad.id}_{elem.lane_section.idx}_{suffix}_{elem.id}"

    def load(self):
        # Parse the xml definition into an initial representation
        start = time.time()
        with open(self._xodr_file, "r") as f:
            od: OpenDriveElement = parse_opendrive(etree.parse(f).getroot())
        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Parsing .xodr file: {elapsed} ms")

        # First pass: create all Road and Lane objects
        start = time.time()
        for road_elem in od.roads:
            road_elem: RoadElement = road_elem

            # Create new road for each lane section
            for section_elem in road_elem.lanes.lane_sections:
                section_elem: LaneSectionElement = section_elem

                # Create new roads so that all lanes for each road are in same direction
                for sub_road, suffix in [
                    (section_elem.leftLanes, "L"),
                    (section_elem.rightLanes, "R"),
                ]:
                    # Skip if there are no lanes
                    if not sub_road:
                        continue

                    road_id = OpenDriveRoadNetwork._elem_id(section_elem, suffix)
                    road = OpenDriveRoadNetwork.Road(
                        road_id,
                        section_elem.parentRoad.junction is not None,
                        section_elem.length,
                        section_elem.sPos,
                    )

                    self._roads[road_id] = road
                    assert road_id not in self._surfaces
                    self._surfaces[road_id] = road

                    for lane_elem in sub_road:
                        lane_id = OpenDriveRoadNetwork._elem_id(lane_elem, suffix)
                        lane = OpenDriveRoadNetwork.Lane(
                            self,
                            lane_id,
                            road,
                            lane_elem.id,
                            section_elem.length,
                            lane_elem.type.lower()
                            in [
                                "driving",
                                "exit",
                                "entry",
                                "offramp",
                                "onramp",
                                "connectingramp",
                            ],
                            road_elem.planView,
                        )
                        # Set road as drivable if it has at least one lane drivable
                        if not road.is_drivable:
                            road.is_drivable = lane_elem.type == "driving"

                        self._lanes[lane_id] = lane
                        self._all_lanes.append(lane)
                        assert lane_id not in self._surfaces
                        self._surfaces[lane_id] = lane
                        road.lanes.append(lane)
        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"First pass: {elapsed} ms")

        # Second pass: compute road and lane connections, compute lane boundaries and polygon
        start = time.time()
        for road_elem in od.roads:
            for section_elem in road_elem.lanes.lane_sections:
                for sub_road, suffix in [
                    (section_elem.leftLanes, "L"),
                    (section_elem.rightLanes, "R"),
                ]:
                    # Skip if there are no lanes
                    if not sub_road:
                        continue

                    road_id = OpenDriveRoadNetwork._elem_id(section_elem, suffix)
                    road: RoadMap.Road = self._roads[road_id]
                    road.bounding_box = [
                        (float("inf"), float("inf")),
                        (float("-inf"), float("-inf")),
                    ]

                    inner_boundary = LaneBoundary(
                        road_elem.planView, None, [], road_elem.lanes.laneOffsets
                    )
                    for lane_elem in sub_road:
                        lane_id = OpenDriveRoadNetwork._elem_id(lane_elem, suffix)
                        lane = self._lanes[lane_id]

                        # Compute lanes in same direction
                        lane.lanes_in_same_direction = [
                            l for l in road.lanes if l.lane_id != lane.lane_id
                        ]

                        # Lanes with positive lane_elem ID run on the left side of the center lane, while lanes with
                        # lane_elem negative ID run on the right side of the center lane.
                        # OpenDRIVE's assumption is that the direction of reference line is same as direction of lanes with
                        # lane_elem negative ID, hence for a given road -1 will be the left most lane in one direction
                        # and 1 will be the left most lane in other direction if it exist.
                        # If there is only one lane in a road, its index will be -1.

                        # Compute lane to the left
                        result = None
                        direction = True
                        if lane.index == -1:
                            left_road_id = road.road_id.replace("R", "L")
                            if left_road_id in self._roads:
                                road_to_left = self._roads[left_road_id]
                                result = road_to_left.lane_at_index(1)
                                direction = False
                        elif lane.index == 1:
                            left_road_id = road.road_id.replace("L", "R")
                            if left_road_id in self._roads:
                                road_to_left = self._roads[left_road_id]
                                result = road_to_left.lane_at_index(-1)
                                direction = False
                        else:
                            assert lane.index != 0
                            result = road.lane_at_index(
                                lane.index - np.sign(lane.index)
                            )
                        lane.lane_to_left = result, direction

                        # Compute lane to right
                        result = None
                        assert abs(lane.index <= len(road.lanes))
                        if abs(lane.index) < len(road.lanes):
                            assert lane.index != 0
                            result = road.lane_at_index(
                                lane.index + np.sign(lane.index)
                            )
                        lane.lane_to_right = result, True

                        # Compute Lane connections
                        self._compute_lane_connections(od, lane, lane_elem, road_elem)

                        # Set lane's outer and inner boundary
                        outer_boundary = LaneBoundary(
                            None,
                            inner_boundary,
                            lane_elem.widths,
                            road_elem.lanes.laneOffsets,
                        )

                        lane.cache_geometry(inner_boundary, outer_boundary)
                        inner_boundary = outer_boundary

                        x_coordinates, y_coordinates = zip(*lane.lane_polygon)
                        lane.bounding_box = [
                            (min(x_coordinates), min(y_coordinates)),
                            (max(x_coordinates), max(y_coordinates)),
                        ]

                        road.bounding_box = [
                            (
                                min(road.bounding_box[0][0], lane.bounding_box[0][0]),
                                min(road.bounding_box[0][1], lane.bounding_box[0][1]),
                            ),
                            (
                                max(road.bounding_box[1][0], lane.bounding_box[1][0]),
                                max(road.bounding_box[1][1], lane.bounding_box[1][1]),
                            ),
                        ]
        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Second pass: {elapsed} ms")

        # Third pass: everything that depends on lane connections
        start = time.time()
        for road in list(self._roads.values()):
            # Compute incoming/outgoing roads based on lane connections
            in_roads = set()
            out_roads = set()
            for lane in road.lanes:
                for in_lane in lane.incoming_lanes:
                    if in_lane.road.road_id != road.road_id:
                        in_roads.add(in_lane.road)
                for out_lane in lane.outgoing_lanes:
                    if out_lane.road.road_id != road.road_id:
                        out_roads.add(out_lane.road)
            road.incoming_roads.extend(list(in_roads))
            road.outgoing_roads.extend(list(out_roads))

            for lane in road.lanes:
                # Compute lane foes
                result = [
                    incoming
                    for outgoing in lane.outgoing_lanes
                    for incoming in outgoing.incoming_lanes
                    if incoming != lane
                ]
                if lane.in_junction:
                    in_roads = set(il.road for il in lane.incoming_lanes)
                    for foe in lane.road.lanes:
                        foe_in_roads = set(il.road for il in foe.incoming_lanes)
                        if not bool(in_roads & foe_in_roads):
                            result.append(foe)
                lane.foes = list(set(result))

        end = time.time()
        elapsed = round((end - start) * 1000.0, 3)
        self._log.info(f"Third pass: {elapsed} ms")

    def _compute_lane_connections(
        self,
        od: OpenDriveElement,
        lane: RoadMap.Lane,
        lane_elem: LaneElement,
        road_elem: RoadElement,
    ):
        lane_link = lane_elem.link
        ls_index = lane_elem.lane_section.idx

        if lane_link.predecessorId:
            road_id, section_id = None, None
            if ls_index == 0:
                # This is the first lane section, so get the first/last lane section of the predecessor road
                road_predecessor = road_elem.link.predecessor
                if road_predecessor and road_predecessor.elementType == "road":
                    road_id = road_predecessor.element_id
                    pred_road_elem = od.getRoad(road_id)
                    section_id = (
                        pred_road_elem.lanes.getLastLaneSectionIdx()
                        if road_predecessor.contactPoint == "end"
                        else 0
                    )
            else:
                # Otherwise, get the previous lane section of the current road
                road_id = road_elem.id
                section_id = ls_index - 1
            if road_id is not None and section_id is not None:
                pred_suffix = "L" if lane_link.predecessorId > 0 else "R"
                pred_lane_id = (
                    f"{road_id}_{section_id}_{pred_suffix}_{lane_link.predecessorId}"
                )
                pred_lane = self.lane_by_id(pred_lane_id)

                if lane.index < 0:
                    # Direction of lane is the same as the reference line
                    if pred_lane not in lane.incoming_lanes:
                        lane.incoming_lanes.append(pred_lane)
                    if lane not in pred_lane.outgoing_lanes:
                        pred_lane.outgoing_lanes.append(lane)
                else:
                    # Direction of lane is opposite the refline, so this is actually an outgoing lane
                    if pred_lane not in lane.outgoing_lanes:
                        lane.outgoing_lanes.append(pred_lane)
                    if lane not in pred_lane.incoming_lanes:
                        pred_lane.incoming_lanes.append(lane)

        if lane_link.successorId:
            road_id, section_id = None, None
            if ls_index == len(road_elem.lanes.lane_sections) - 1:
                # This is the last lane section, so get the first/last lane section of the successor road
                road_successor = road_elem.link.successor
                if road_successor and road_successor.elementType == "road":
                    road_id = road_successor.element_id
                    succ_road_elem = od.getRoad(road_id)
                    section_id = (
                        succ_road_elem.lanes.getLastLaneSectionIdx()
                        if road_successor.contactPoint == "end"
                        else 0
                    )
            else:
                # Otherwise, get the next lane section in the current road
                road_id = road_elem.id
                section_id = ls_index + 1

            if road_id is not None and section_id is not None:
                succ_suffix = "L" if lane_link.successorId > 0 else "R"
                succ_lane_id = (
                    f"{road_id}_{section_id}_{succ_suffix}_{lane_link.successorId}"
                )
                succ_lane = self.lane_by_id(succ_lane_id)

                if lane.index < 0:
                    # Direction of lane is the same as the reference line
                    if succ_lane not in lane.outgoing_lanes:
                        lane.outgoing_lanes.append(succ_lane)
                    if lane not in succ_lane.incoming_lanes:
                        succ_lane.incoming_lanes.append(lane)
                else:
                    # Direction of lane is opposite the refline, so this is actually an incoming lane
                    if succ_lane not in lane.incoming_lanes:
                        lane.incoming_lanes.append(succ_lane)
                    if lane not in succ_lane.outgoing_lanes:
                        succ_lane.outgoing_lanes.append(lane)

    @property
    def source(self) -> str:
        """This is the .xodr file of the OpenDRIVE map."""
        return self._xodr_file

    def surface_by_id(self, surface_id: str) -> RoadMap.Surface:
        return self._surfaces.get(surface_id)

    @cached_property
    def bounding_box(self) -> BoundingBox:
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for road_id in self._roads:
            road = self._roads[road_id]
            x_mins.append(road.bounding_box[0][0])
            y_mins.append(road.bounding_box[0][1])
            x_maxs.append(road.bounding_box[1][0])
            y_maxs.append(road.bounding_box[1][1])

        return BoundingBox(
            min_pt=Point(x=min(x_mins), y=min(y_mins)),
            max_pt=Point(x=max(x_maxs), y=max(y_maxs)),
        )

    def to_glb(self, at_path):
        """build a glb file for camera rendering and envision"""
        glb = self._make_glb_from_polys()
        glb.write_glb(at_path)

    def _make_glb_from_polys(self):
        scene = trimesh.Scene()
        polygons = []
        for lane_id in self._lanes:
            lane = self._lanes[lane_id]
            polygons.append(lane.shape())

        mesh = generate_mesh_from_polygons(polygons)

        # Attach additional information for rendering as metadata in the map glb
        # <2D-BOUNDING_BOX>: four floats separated by ',' (<FLOAT>,<FLOAT>,<FLOAT>,<FLOAT>),
        # which describe x-minimum, y-minimum, x-maximum, and y-maximum
        metadata = {
            "bounding_box": (
                self.bounding_box.min_pt.x,
                self.bounding_box.min_pt.y,
                self.bounding_box.max_pt.x,
                self.bounding_box.max_pt.y,
            )
        }

        # lane markings information
        lane_dividers, road_dividers = self._compute_traffic_dividers()
        metadata["lane_dividers"] = lane_dividers
        metadata["edge_dividers"] = road_dividers

        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial()
        )

        scene.add_geometry(mesh)
        return _GLBData(gltf.export_glb(scene, extras=metadata, include_normals=True))

    def _compute_traffic_dividers(self, threshold=1):
        lane_dividers = []  # divider between lanes with same traffic direction
        road_dividers = []  # divider between roads with opposite traffic direction
        dividers_checked = []
        for road_id in self._roads:
            road = self._roads[road_id]
            road_left_border = None
            for lane in road.lanes:
                left_border_vertices_len = int((len(lane.lane_polygon) - 1) / 2)
                left_side = lane.lane_polygon[:left_border_vertices_len]
                if lane.index not in [1, -1]:
                    lane_dividers.append(left_side)
                else:
                    road_left_border = left_side

            assert road_left_border

            # The road borders that overlapped in positions form an edge divider
            id_split = road_id.split("_")
            parent_road_id = f"{id_split[0]}_{id_split[1]}"
            if parent_road_id not in dividers_checked:
                dividers_checked.append(parent_road_id)
                if "R" in road.road_id:
                    adjacent_road_id = road.road_id.replace("R", "L")
                else:
                    adjacent_road_id = road.road_id.replace("L", "R")
                if adjacent_road_id in self._roads:
                    road_dividers.append(road_left_border)

        return lane_dividers, road_dividers

    class Surface(RoadMap.Surface):
        def __init__(self, surface_id: str):
            self._surface_id = surface_id

        @property
        def surface_id(self) -> str:
            return self._surface_id

        @property
        def is_drivable(self) -> bool:
            # Not all lanes on OpenDRIVE roads are drivable
            raise NotImplementedError

    class Lane(RoadMap.Lane, Surface):
        def __init__(
            self,
            road_map,
            lane_id: str,
            road: RoadMap.Road,
            index: int,
            length: float,
            is_drivable: bool,
            road_plan_view: PlanViewElement,
        ):
            super().__init__(lane_id)
            self._map = road_map
            self._lane_id = lane_id
            self._road = road
            self._index = index
            self._length = length
            self._plan_view = road_plan_view
            self._is_drivable = is_drivable
            self._incoming_lanes = []
            self._outgoing_lanes = []
            self._lanes_in_same_dir = []
            self._foes = []
            self._ref_coords = {}
            self._lane_boundaries = tuple()
            self._lane_polygon = []
            self._centerline_points = []
            self._bounding_box = []
            self._lane_to_left = None, True
            self._lane_to_right = None, True
            self._in_junction = None

        @property
        def is_drivable(self) -> bool:
            return self._is_drivable

        @property
        def lane_id(self) -> str:
            return self._lane_id

        @property
        def road(self) -> RoadMap.Road:
            return self._road

        @property
        def length(self) -> float:
            return self._length

        @property
        def in_junction(self) -> bool:
            return self.road.is_junction

        @property
        def index(self) -> int:
            # TODO: convert to expected convention?
            return self._index

        @property
        def incoming_lanes(self) -> List[RoadMap.Lane]:
            return self._incoming_lanes

        @property
        def outgoing_lanes(self) -> List[RoadMap.Lane]:
            return self._outgoing_lanes

        @property
        def entry_surfaces(self) -> List[RoadMap.Surface]:
            return self.incoming_lanes

        @property
        def exit_surfaces(self) -> List[RoadMap.Surface]:
            return self.outgoing_lanes

        @property
        def lanes_in_same_direction(self) -> List[RoadMap.Lane]:
            return self._lanes_in_same_dir

        @lanes_in_same_direction.setter
        def lanes_in_same_direction(self, lanes):
            self._lanes_in_same_dir = lanes

        @property
        def lane_to_left(self) -> Tuple[RoadMap.Lane, bool]:
            return self._lane_to_left

        @lane_to_left.setter
        def lane_to_left(self, value):
            self._lane_to_left = value

        @property
        def lane_to_right(self) -> Tuple[RoadMap.Lane, bool]:
            return self._lane_to_right

        @lane_to_right.setter
        def lane_to_right(self, value):
            self._lane_to_right = value

        @property
        def foes(self) -> List[RoadMap.Lane]:
            return self._foes

        @foes.setter
        def foes(self, value):
            self._foes = value

        @property
        def lane_polygon(self) -> List[Tuple[float, float]]:
            return self._lane_polygon

        # Central Reference line of the lane, (For vector and heading computation)
        @property
        def centerline_points(self) -> List[Tuple[float, float]]:
            return self._centerline_points

        @property
        def bounding_box(self) -> List[Tuple[float, float]]:
            return self._bounding_box

        @bounding_box.setter
        def bounding_box(self, value):
            self._bounding_box = value

        def t_angle(self, s_heading: float):
            lane_elem_id = self._index
            angle = (
                (s_heading - math.pi / 2)
                if lane_elem_id < 0
                else (s_heading + math.pi / 2)
            )
            return constrain_angle(angle)

        def cache_geometry(
            self, inner_boundary: LaneBoundary, outer_boundary: LaneBoundary
        ):
            # Set inner/outer boundaries
            self._lane_boundaries = (inner_boundary, outer_boundary)

            # Compute ref coords (s values to sample for polygon & centerline)
            section_len = self._length
            section_s_start = self.road.s_pos
            section_s_end = section_s_start + section_len

            inner_s_vals = inner_boundary.to_linear_segments(
                section_s_start, section_s_end
            )
            outer_s_vals = outer_boundary.to_linear_segments(
                section_s_start, section_s_end
            )

            # Cache centerline & ref coords
            center_xs, center_ys = [], []
            s_vals = sorted(set(inner_s_vals + outer_s_vals))
            for s in s_vals:
                t_inner = inner_boundary.calc_t(s, section_s_start, self.index)
                t_outer = outer_boundary.calc_t(s, section_s_start, self.index)
                (x_ref, y_ref), heading = self._plan_view.calc(s)
                angle = self.t_angle(heading)
                width_at_offset = t_outer - t_inner
                center_xs.append(
                    x_ref + (t_inner + (width_at_offset / 2)) * math.cos(angle)
                )
                center_ys.append(
                    y_ref + (t_inner + (width_at_offset / 2)) * math.sin(angle)
                )
                self._ref_coords[s] = (t_inner, t_outer)

            # For lanes left of the refline, reverse the order of centerline points to be in order of increasing s
            if self.index > 0:
                center_xs = center_xs[::-1]
                center_ys = center_ys[::-1]
            self._centerline_points = list(zip(center_xs, center_ys))

            # Cache lane polygon (normal size, with no buffer)
            self._lane_polygon = self.compute_lane_polygon()

        def compute_lane_polygon(
            self,
            width_offset: float = 0.0,
        ) -> List[Tuple[float, float]]:
            xs, ys = [], []
            xs_inner, ys_inner = [], []
            xs_outer, ys_outer = [], []
            s_vals = sorted(self._ref_coords.keys())
            for s in s_vals:
                t_inner, t_outer = self._ref_coords[s]
                (x_ref, y_ref), heading = self._plan_view.calc(s)
                angle = self.t_angle(heading)
                xs_inner.append(x_ref + (t_inner - width_offset) * math.cos(angle))
                ys_inner.append(y_ref + (t_inner - width_offset) * math.sin(angle))
                xs_outer.append(x_ref + (t_outer + width_offset) * math.cos(angle))
                ys_outer.append(y_ref + (t_outer + width_offset) * math.sin(angle))
            xs.extend(xs_inner + xs_outer[::-1] + [xs_inner[0]])
            ys.extend(ys_inner + ys_outer[::-1] + [ys_inner[0]])
            return list(zip(xs, ys))

        @lru_cache(maxsize=8)
        def project_along(
            self, start_offset: float, distance: float
        ) -> Set[Tuple[RoadMap.Lane, float]]:
            return super().project_along(start_offset, distance)

        @lru_cache(maxsize=8)
        def contains_point(self, point: Point) -> bool:
            if (
                self._bounding_box[0][0] <= point[0] <= self._bounding_box[1][0]
                and self._bounding_box[0][1] <= point[1] <= self._bounding_box[1][1]
            ):
                lane_point = self.to_lane_coord(point)
                width_at_offset = self.width_at_offset(lane_point.s)
                # t-direction is negative for right side and positive for left side of the central reference
                # line of lane w.r.t its heading, absolute value of lane_point.t should be less than half of width at
                # that point
                return (
                    abs(lane_point.t) <= width_at_offset
                    and 0 <= lane_point.s < self.length
                )
            return False

        @lru_cache(maxsize=8)
        def offset_along_lane(self, world_point: Point) -> float:
            return offset_along_shape(world_point[:2], self._centerline_points)

        @lru_cache(maxsize=16)
        def oncoming_lanes_at_offset(self, offset: float) -> List[RoadMap.Lane]:
            result = []
            radius = 1.1 * self.width_at_offset(offset)
            pt = self.from_lane_coord(RefLinePoint(offset))
            nearby_lanes = self._map.nearest_lanes(pt, radius=radius)
            if not nearby_lanes:
                return result
            my_vect = self.vector_at_offset(offset)
            my_norm = np.linalg.norm(my_vect)
            threshold = -0.995562  # cos(175*pi/180)
            for lane, _ in nearby_lanes:
                if lane == self:
                    continue
                lane_refline_pt = lane.to_lane_coord(pt)
                lv = lane.vector_at_offset(lane_refline_pt.s)
                lane_angle = np.dot(my_vect, lv) / (my_norm * np.linalg.norm(lv))
                if lane_angle < threshold:
                    result.append(lane)
            return result

        @lru_cache(maxsize=8)
        def from_lane_coord(self, lane_point: RefLinePoint) -> Point:
            x, y = position_at_shape_offset(self._centerline_points, lane_point.s)
            return Point(x=x, y=y)

        @lru_cache(maxsize=8)
        def to_lane_coord(self, world_point: Point) -> RefLinePoint:
            return super().to_lane_coord(world_point)

        @lru_cache(maxsize=8)
        def center_at_point(self, point: Point) -> Point:
            return super().center_at_point(point)

        @lru_cache(8)
        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            reference_line_vertices_len = int((len(self._lane_polygon) - 1) / 2)
            # left_edge
            left_edge_shape = self._lane_polygon[:reference_line_vertices_len]
            left_offset = offset_along_shape(point[:2], left_edge_shape)
            x, y = position_at_shape_offset(left_edge_shape, left_offset)
            left_edge = Point(x, y)

            # right_edge
            right_edge_shape = self._lane_polygon[
                reference_line_vertices_len : len(self._lane_polygon) - 1
            ]
            right_offset = offset_along_shape(point[:2], right_edge_shape)
            x, y = position_at_shape_offset(right_edge_shape, right_offset)
            right_edge = Point(x, y)
            return left_edge, right_edge

        @lru_cache(8)
        def vector_at_offset(self, start_offset: float) -> np.ndarray:
            return super().vector_at_offset(start_offset)

        @lru_cache(maxsize=8)
        def center_pose_at_point(self, point: Point) -> Pose:
            return super().center_pose_at_point(point)

        @lru_cache(maxsize=8)
        def curvature_radius_at_offset(
            self, offset: float, lookahead: int = 5
        ) -> float:
            return super().curvature_radius_at_offset(offset, lookahead)

        def width_at_offset(self, lane_point_s: float) -> float:
            if self.index < 0:
                road_offset = lane_point_s + self.road.s_pos
            else:
                road_offset = (self._length - lane_point_s) + self.road.s_pos
            inner_boundary, outer_boundary = self._lane_boundaries
            t_outer = outer_boundary.calc_t(road_offset, self.road.s_pos, self.index)
            t_inner = inner_boundary.calc_t(road_offset, self.road.s_pos, self.index)
            return abs(t_outer - t_inner)

        @lru_cache(maxsize=4)
        def shape(self, width: float = 0.0, buffer_width: float = 0.0) -> Polygon:
            if buffer_width == 0.0:
                return Polygon(self._lane_polygon)
            buffered_polygon = self.compute_lane_polygon(buffer_width / 2)
            return Polygon(buffered_polygon)

    def lane_by_id(self, lane_id: str) -> RoadMap.Lane:
        lane = self._lanes.get(lane_id)
        if not lane:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown lane_id '{lane_id}'"
            )
        return lane

    class Road(RoadMap.Road, Surface):
        def __init__(
            self,
            road_id: str,
            is_junction: bool,
            length: float,
            s_pos: float,
        ):
            super().__init__(road_id)
            self._log = logging.getLogger(self.__class__.__name__)
            self._road_id = road_id
            self._is_junction = is_junction
            self._length = length
            self._s_pos = s_pos
            self._is_drivable = False
            self._lanes = []
            self._bounding_box = []
            self._incoming_roads = []
            self._outgoing_roads = []
            self._parallel_roads = []

        @property
        def road_id(self) -> str:
            return self._road_id

        @property
        def is_junction(self) -> bool:
            return self._is_junction

        @property
        def length(self) -> float:
            return self._length

        @property
        def s_pos(self) -> float:
            return self._s_pos

        @property
        def is_drivable(self) -> bool:
            return self._is_drivable

        @is_drivable.setter
        def is_drivable(self, value):
            self._is_drivable = value

        @property
        def incoming_roads(self) -> List[RoadMap.Road]:
            return self._incoming_roads

        @property
        def outgoing_roads(self) -> List[RoadMap.Road]:
            return self._outgoing_roads

        @property
        def entry_surfaces(self) -> List[RoadMap.Surface]:
            # TAI:  also include lanes here?
            return self.incoming_roads

        @property
        def exit_surfaces(self) -> List[RoadMap.Surface]:
            # TAI:  also include lanes here?
            return self.outgoing_roads

        @property
        def parallel_roads(self) -> List[RoadMap.Road]:
            return self._parallel_roads

        @property
        def lanes(self) -> List[RoadMap.Lane]:
            return self._lanes

        @property
        def bounding_box(self) -> List[Tuple[float, float]]:
            return self._bounding_box

        @bounding_box.setter
        def bounding_box(self, value):
            self._bounding_box = value

        @lru_cache(maxsize=8)
        def contains_point(self, point: Point) -> bool:
            if (
                self._bounding_box[0][0] <= point[0] <= self._bounding_box[1][0]
                and self._bounding_box[0][1] <= point[1] <= self._bounding_box[1][1]
            ):
                for lane in self.lanes:
                    if lane.contains_point(point):
                        return True
            return False

        @lru_cache(maxsize=8)
        def edges_at_point(self, point: Point) -> Tuple[Point, Point]:
            # left and right edge follow the lane reference line system or direction of that road
            leftmost_lane, rightmost_lane = None, None
            min_index, max_index = float("inf"), float("-inf")
            for lane in self.lanes:
                if abs(lane.index) < min_index:
                    min_index = abs(lane.index)
                    leftmost_lane = lane
                if abs(lane.index) > max_index:
                    max_index = abs(lane.index)
                    rightmost_lane = lane
            _, right_edge = rightmost_lane.edges_at_point(point)
            left_edge, _ = leftmost_lane.edges_at_point(point)

            return left_edge, right_edge

        @lru_cache(maxsize=16)
        def oncoming_roads_at_point(self, point: Point) -> List[RoadMap.Road]:
            return super().oncoming_roads_at_point(point)

        @lru_cache(maxsize=4)
        def shape(self, width: float = 0.0, buffer_width: float = 0.0) -> Polygon:
            leftmost_lane, rightmost_lane = None, None
            min_index, max_index = float("inf"), float("-inf")
            for lane in self.lanes:
                if abs(lane.index) < min_index:
                    min_index = abs(lane.index)
                    leftmost_lane = lane
                if abs(lane.index) > max_index:
                    max_index = abs(lane.index)
                    rightmost_lane = lane

            if buffer_width == 0.0:
                rightmost_lane_buffered_polygon = rightmost_lane.lane_polygon
                leftmost_lane_buffered_polygon = leftmost_lane.lane_polygon
            else:
                rightmost_lane_buffered_polygon = rightmost_lane.compute_lane_polygon(
                    buffer_width
                )
                leftmost_lane_buffered_polygon = leftmost_lane.compute_lane_polygon(
                    buffer_width
                )

            # Right edge
            rightmost_edge_vertices_len = int(
                (len(rightmost_lane_buffered_polygon) - 1) / 2
            )
            rightmost_edge_shape = rightmost_lane_buffered_polygon[
                rightmost_edge_vertices_len : len(rightmost_lane_buffered_polygon) - 1
            ]

            # Left edge
            leftmost_edge_vertices_len = int(
                (len(leftmost_lane_buffered_polygon) - 1) / 2
            )
            leftmost_edge_shape = leftmost_lane_buffered_polygon[
                :leftmost_edge_vertices_len
            ]

            road_polygon = (
                leftmost_edge_shape + rightmost_edge_shape + [leftmost_edge_shape[0]]
            )
            return Polygon(road_polygon)

        def lane_at_index(self, index: int) -> RoadMap.Lane:
            lanes_with_index = [lane for lane in self.lanes if lane.index == index]
            if len(lanes_with_index) == 0:
                self._log.warning(
                    f"Road with id {self.road_id} has no lane at index {index}"
                )
                return None
            assert len(lanes_with_index) == 1
            return lanes_with_index[0]

    def road_by_id(self, road_id: str) -> RoadMap.Road:
        road = self._roads.get(road_id)
        if not road:
            self._log.warning(
                f"OpenDriveRoadNetwork got request for unknown road_id '{road_id}'"
            )
        return road

    def _build_lane_r_tree(self):
        result = rtree.index.Index()
        result.interleaved = True
        for index, lane in enumerate(self._all_lanes):
            bounding_box = (
                lane.bounding_box[0][0],
                lane.bounding_box[0][1],
                lane.bounding_box[1][0],
                lane.bounding_box[1][1],
            )
            result.add(index, bounding_box)
        return result

    def _get_neighboring_lanes(self, x, y, r=0.1):
        neighboring_lanes = []

        if self._lane_rtree is None:
            self._lane_rtree = self._build_lane_r_tree()

        assert len(self._all_lanes) != 0
        for i in self._lane_rtree.intersection((x - r, y - r, x + r, y + r)):
            lane = self._all_lanes[i]
            d = distance_point_to_polygon((x, y), lane.lane_polygon)
            if d < r:
                neighboring_lanes.append((lane, d))
        return neighboring_lanes

    @lru_cache(maxsize=16)
    def nearest_lanes(
        self, point: Point, radius: float = None, include_junctions=True
    ) -> List[Tuple[RoadMap.Lane, float]]:
        if radius is None:
            radius = max(10, 2 * self._default_lane_width)
        candidate_lanes = self._get_neighboring_lanes(point[0], point[1], r=radius)
        candidate_lanes.sort(key=lambda lane_dist_tup: lane_dist_tup[1])
        return candidate_lanes

    def nearest_lane(
        self, point: Point, radius: float = None, include_junctions=True
    ) -> RoadMap.Lane:
        nearest_lanes = self.nearest_lanes(point, radius, include_junctions)
        for lane, dist in nearest_lanes:
            if lane.contains_point(point):
                return lane
        return nearest_lanes[0][0] if nearest_lanes else None

    @lru_cache(maxsize=16)
    def road_with_point(self, point: Point) -> RoadMap.Road:
        radius = max(5, 2 * self._default_lane_width)
        for nl, dist in self.nearest_lanes(point, radius):
            if nl.contains_point(point):
                return nl.road
        return None

    @staticmethod
    def _shortest_route(start: RoadMap.Road, end: RoadMap.Road) -> List[RoadMap.Road]:
        queue = [(start.length, start.road_id, start)]
        came_from = dict()
        came_from[start] = None
        cost_so_far = dict()
        cost_so_far[start] = start.length
        current = None

        # Dijkstra’s Algorithm
        while queue:
            (_, _, current) = heapq.heappop(queue)
            current: RoadMap.Road
            if current == end:
                break
            for out_road in current.outgoing_roads:
                new_cost = cost_so_far[current] + out_road.length
                if out_road not in cost_so_far or new_cost < cost_so_far[out_road]:
                    cost_so_far[out_road] = new_cost
                    came_from[out_road] = current
                    heapq.heappush(queue, (new_cost, out_road.road_id, out_road))

        # This means we couldn't find a valid route since the queue is empty
        if current != end:
            return []

        # Reconstruct path
        current = end
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start)
        path.reverse()
        return path

    def generate_routes(
        self,
        start_road: RoadMap.Road,
        end_road: RoadMap.Road,
        via: Sequence[RoadMap.Road] = None,
        max_to_gen: int = 1,
    ) -> List[RoadMap.Route]:
        assert (
            max_to_gen == 1
        ), "multiple route generation not yet supported for OpenDRIVE"
        newroute = OpenDriveRoadNetwork.Route(self)
        result = [newroute]

        roads = [start_road]
        if via:
            roads += via
        if end_road != start_road:
            roads.append(end_road)

        route_roads = []
        for cur_road, next_road in zip(roads, roads[1:] + [None]):
            if not next_road:
                route_roads.append(cur_road)
                break
            sub_route = OpenDriveRoadNetwork._shortest_route(cur_road, next_road) or []
            if len(sub_route) < 2:
                self._log.warning(
                    f"Unable to find valid path between {(cur_road.road_id, next_road.road_id)}."
                )
                return result
            # The sub route includes the boundary roads (cur_road, next_road).
            # We clip the latter to prevent duplicates
            route_roads.extend(sub_route[:-1])

        for road in route_roads:
            newroute.add_road(road)
        return result

    def random_route(self, max_route_len: int = 10) -> RoadMap.Route:
        route = OpenDriveRoadNetwork.Route(self)
        next_roads = list(self._roads.values())
        while next_roads and len(route.roads) < max_route_len:
            cur_road = random.choice(next_roads)
            route.add_road(cur_road)
            next_roads = list(cur_road.outgoing_roads)
        return route

    def empty_route(self) -> RoadMap.Route:
        return OpenDriveRoadNetwork.Route(self)

    class Route(RoadMap.Route):
        def __init__(self, road_map):
            self._roads = []
            self._length = 0
            self._map = road_map

        @property
        def roads(self) -> List[RoadMap.Road]:
            return self._roads

        @property
        def road_length(self) -> float:
            return self._length

        def add_road(self, road: RoadMap.Road):
            self._length += road.length
            self._roads.append(road)

        @cached_property
        def geometry(self) -> Sequence[Sequence[Tuple[float, float]]]:
            return [list(road.shape(0.0, 1.0).exterior.coords) for road in self.roads]

        @lru_cache(maxsize=8)
        def distance_between(self, start: Point, end: Point) -> float:
            for cand_start_lane, _ in self._map.nearest_lanes(start, 30.0, False):
                try:
                    sind = self._roads.index(cand_start_lane.road)
                    break
                except ValueError:
                    pass
            else:
                logging.warning("unable to find road on route near start point")
                return None
            start_road = cand_start_lane.road
            for cand_end_lane, _ in self._map.nearest_lanes(end, 30.0, False):
                try:
                    eind = self._roads.index(cand_end_lane.road)
                    break
                except ValueError:
                    pass
            else:
                logging.warning("unable to find road on route near end point")
                return None
            end_road = cand_end_lane.road
            d = 0
            start_offset = cand_start_lane.offset_along_lane(start)
            end_offset = cand_end_lane.offset_along_lane(end)
            if start_road == end_road:
                return end_offset - start_offset
            negate = False
            if sind > eind:
                cand_start_lane = cand_end_lane
                start_road, end_road = end_road, start_road
                start_offset, end_offset = end_offset, start_offset
                negate = True
            for road in self._roads:
                if d == 0 and road == start_road:
                    d += cand_start_lane.length - start_offset
                elif road == end_road:
                    d += end_offset
                    break
                elif d > 0:
                    d += road.length
            return -d if negate else d

        @lru_cache(maxsize=8)
        def project_along(
            self, start: Point, distance: float
        ) -> Set[Tuple[RoadMap.Lane, float]]:
            route_roads = set(self._roads)
            for cand_start_lane, _ in self._map.nearest_lanes(start, 30.0, False):
                if cand_start_lane.road in route_roads:
                    break
            else:
                logging.warning("unable to find road on route near start point")
                return None
            started = False
            for road in self._roads:
                if not started:
                    if road != cand_start_lane.road:
                        continue
                    started = True
                    lane_pt = cand_start_lane.to_lane_coord(start)
                    start_offset = lane_pt.s
                else:
                    start_offset = 0
                if distance > road.length - start_offset:
                    distance -= road.length - start_offset
                    continue
                return {(lane, distance) for lane in road.lanes}
            return set()
