# MIT License
#
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
import math
from os import path
from pathlib import Path
import pytest
from shapely.geometry.polygon import Polygon
from smarts.core.coordinates import Point
from smarts.core.opendrive_road_network import OpenDriveRoadNetwork
from smarts.core.scenario import Scenario
from smarts.core.sumo_road_network import SumoRoadNetwork


@pytest.fixture
def sumo_scenario():
    return Scenario(scenario_root="scenarios/intersections/4lane")


@pytest.fixture
def opendrive_scenario():
    return Scenario(scenario_root="scenarios/opendrive")


def test_sumo_map(sumo_scenario):
    road_map = sumo_scenario.road_map
    assert isinstance(road_map, SumoRoadNetwork)

    point = (125.20, 139.0, 0)
    lane = road_map.nearest_lane(point)
    assert lane.lane_id == "edge-north-NS_0"
    assert lane.road.road_id == "edge-north-NS"
    assert lane.index == 0
    assert lane.road.contains_point(point)
    assert lane.is_drivable
    assert len(lane.shape()) >= 2

    right_lane, direction = lane.lane_to_right
    assert not right_lane

    left_lane, direction = lane.lane_to_left
    assert left_lane
    assert direction
    assert left_lane.lane_id == "edge-north-NS_1"
    assert left_lane.index == 1

    lefter_lane, direction = left_lane.lane_to_left
    assert not lefter_lane

    on_roads = lane.road.oncoming_roads_at_point(point)
    assert on_roads
    assert len(on_roads) == 1
    assert on_roads[0].road_id == "edge-north-SN"

    reflinept = lane.to_lane_coord(point)
    assert reflinept.s == 1.0
    assert reflinept.t == 0.0

    offset = reflinept.s
    assert lane.width_at_offset(offset) == 3.2
    assert lane.curvature_radius_at_offset(offset) == math.inf

    on_lanes = lane.oncoming_lanes_at_offset(offset)
    assert not on_lanes
    on_lanes = left_lane.oncoming_lanes_at_offset(offset)
    assert len(on_lanes) == 1
    assert on_lanes[0].lane_id == "edge-north-SN_1"

    in_lanes = lane.incoming_lanes
    assert not in_lanes

    out_lanes = lane.outgoing_lanes
    assert out_lanes
    assert len(out_lanes) == 2
    assert out_lanes[0].lane_id == ":junction-intersection_0_0"
    assert out_lanes[1].lane_id == ":junction-intersection_1_0"

    foes = out_lanes[0].foes
    assert foes
    assert len(foes) == 3
    foe_set = set(f.lane_id for f in foes)
    assert "edge-east-EW_0" in foe_set  # entering from east
    assert "edge-north-NS_0" in foe_set  # entering from north
    assert ":junction-intersection_5_0" in foe_set  # crossing from east-to-west

    r1 = road_map.road_by_id("edge-north-NS")
    assert r1
    assert r1.is_drivable
    assert len(r1.shape()) >= 2
    r2 = road_map.road_by_id("edge-east-WE")
    assert r2
    assert r2.is_drivable
    assert len(r2.shape()) >= 2

    routes = road_map.generate_routes(r1, r2)
    assert routes
    assert len(routes[0].roads) == 4

    route = routes[0]
    db = route.distance_between(point, (198, 65.20, 0))
    assert db == 134.01

    cands = route.project_along(point, 134.01)
    for r2lane in r2.lanes:
        assert (r2lane, 53.6) in cands

    cands = left_lane.project_along(offset, 134.01)
    assert len(cands) == 6
    for r2lane in r2.lanes:
        if r2lane.index == 1:
            assert any(
                r2lane == cand[0] and math.isclose(cand[1], 53.6) for cand in cands
            )


def test_od_map_junction():
    root = path.join(Path(__file__).parent.absolute(), "maps")
    road_map = OpenDriveRoadNetwork.from_file(
        path.join(root, "UC_Simple-X-Junction.xodr")
    )
    assert isinstance(road_map, OpenDriveRoadNetwork)

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0

    # Road tests
    r_0_R = road_map.road_by_id("0_0_R")
    assert r_0_R
    assert not r_0_R.is_junction
    assert r_0_R.length == 103
    assert len(r_0_R.lanes) == 4
    assert r_0_R.lane_at_index(0) is None
    assert r_0_R.lane_at_index(1) is None
    assert r_0_R.lane_at_index(-1).road.road_id == "0_0_R"
    assert len(r_0_R.shape().exterior.coords) == 5
    assert set([r.road_id for r in r_0_R.incoming_roads]) == {"5_0_R", "7_0_R", "9_0_R"}
    assert set([r.road_id for r in r_0_R.outgoing_roads]) == set()

    r_0_L = road_map.road_by_id("0_0_L")
    assert r_0_L
    assert not r_0_L.is_junction
    assert r_0_L.length == 103
    assert len(r_0_L.lanes) == 4
    assert r_0_L.lane_at_index(0) is None
    assert r_0_L.lane_at_index(-1) is None
    assert r_0_L.lane_at_index(1).road.road_id == "0_0_L"
    assert len(r_0_L.shape().exterior.coords) == 5
    assert set([r.road_id for r in r_0_L.incoming_roads]) == set()
    assert set([r.road_id for r in r_0_L.outgoing_roads]) == {
        "3_0_R",
        "8_0_R",
        "15_0_R",
    }

    r_13_R = road_map.road_by_id("13_0_R")
    assert r_13_R
    assert not r_13_R.is_junction
    assert r_13_R.length == 103
    assert len(r_13_R.lanes) == 4
    assert r_13_R.lane_at_index(0) is None
    assert r_13_R.lane_at_index(1) is None
    assert r_13_R.lane_at_index(-1).road.road_id == "13_0_R"
    assert set([r.road_id for r in r_13_R.incoming_roads]) == {
        "10_0_R",
        "12_0_R",
        "15_0_R",
    }
    assert set([r.road_id for r in r_13_R.outgoing_roads]) == set()

    r_13_L = road_map.road_by_id("13_0_L")
    assert r_13_L
    assert not r_13_L.is_junction
    assert r_13_L.length == 103
    assert len(r_13_L.lanes) == 4
    assert r_13_L.lane_at_index(0) is None
    assert r_13_L.lane_at_index(-1) is None
    assert r_13_L.lane_at_index(1).road.road_id == "13_0_L"
    assert set([r.road_id for r in r_13_L.incoming_roads]) == set()
    assert set([r.road_id for r in r_13_L.outgoing_roads]) == {
        "9_0_R",
        "11_0_R",
        "14_0_R",
    }

    # Lane tests
    l1 = road_map.lane_by_id("0_0_L_1")
    assert l1
    assert l1.road.road_id == "0_0_L"
    assert l1.index == 1
    assert len(l1.lanes_in_same_direction) == 3
    assert l1.length == 103
    assert l1.is_drivable
    assert len(l1.shape().exterior.coords) == 5

    assert [l.lane_id for l in l1.incoming_lanes] == []
    assert [l.lane_id for l in l1.outgoing_lanes] == [
        "3_0_R_-1",
        "8_0_R_-1",
        "15_0_R_-1",
    ]

    right_lane, direction = l1.lane_to_right
    assert right_lane
    assert direction
    assert right_lane.lane_id == "0_0_L_2"
    assert right_lane.index == 2

    further_right_lane, direction = right_lane.lane_to_right
    assert further_right_lane
    assert direction
    assert further_right_lane.lane_id == "0_0_L_3"
    assert further_right_lane.index == 3

    left_lane, direction = l1.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "0_0_R_-1"
    assert left_lane.index == -1

    # point on lane
    point = (118.0, 170.0, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 70.0
    assert round(refline_pt.t, 2) == -0.12

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.75
    assert l1.curvature_radius_at_offset(offset) == math.inf
    assert l1.contains_point(point)
    assert l1.road.contains_point(point)
    central_point = l1.center_at_point(point)
    assert (round(central_point.x, 2), round(central_point.y, 2)) == (118.12, 170.0)

    # oncoming lanes at this point
    on_lanes = l1.oncoming_lanes_at_offset(offset)
    assert on_lanes
    assert len(on_lanes) == 1
    assert on_lanes[0].lane_id == "0_0_R_-1"

    # lane edges on point
    left_edge, right_edge = l1.edges_at_point(point)
    assert (round(left_edge.x, 2), round(left_edge.y, 2)) == (120.0, 170.0)
    assert (round(right_edge.x, 2), round(right_edge.y, 2)) == (116.25, 170.0)

    # road edges on point
    road_left_edge, road_right_edge = r_0_L.edges_at_point(point)
    assert (round(road_left_edge.x, 2), round(road_left_edge.y, 2)) == (120.0, 170.0)
    assert (round(road_right_edge.x, 2), round(road_right_edge.y, 2)) == (109.7, 170.0)

    # check for locations (lane, offset tuples) within distance at this offset
    candidates = l1.project_along(offset, 70)
    assert (len(candidates)) == 11

    # point not on lane but on road
    point = (112.0, 170.0, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 70.0
    assert round(refline_pt.t, 2) == -6.12

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.75
    assert l1.curvature_radius_at_offset(offset) == math.inf
    assert not l1.contains_point(point)
    assert l1.road.contains_point(point)

    l3 = road_map.lane_by_id("9_0_R_-1")
    assert l3
    assert l3.road.road_id == "9_0_R"
    assert l3.index == -1
    assert l3.is_drivable

    foes = l3.foes
    assert set(f.lane_id for f in foes) == {"5_0_R_-1", "7_0_R_-1"}

    # road edges on point for a road with one lane
    point = (115.55, 120.63)
    r5 = road_map.road_by_id("5_0_R")
    road_left_edge, road_right_edge = r5.edges_at_point(point)
    assert (round(road_left_edge.x, 2), round(road_left_edge.y, 2)) == (115.24, 121.23)
    assert (round(road_right_edge.x, 2), round(road_right_edge.y, 2)) == (
        116.99,
        117.91,
    )

    # nearest lane for a point outside road
    point = (109.0, 160.0, 0)
    l4 = road_map.nearest_lane(point)
    assert l4.lane_id == "0_0_L_4"
    assert l4.road.road_id == "0_0_L"
    assert l4.index == 4
    assert not l4.road.contains_point(point)
    assert not l4.is_drivable

    # nearest lane for a point inside road
    point = (117.0, 150.0, 0)
    l5 = road_map.nearest_lane(point)
    assert l5.lane_id == "0_0_L_1"
    assert l5.road.road_id == "0_0_L"
    assert l5.index == 1
    assert l5.road.contains_point(point)
    assert l5.is_drivable

    # route generation
    r_0_0_L = road_map.road_by_id("0_0_L")
    r_13_0_R = road_map.road_by_id("13_0_R")
    r_15_0_R = road_map.road_by_id("15_0_R")

    route_0_to_13 = road_map.generate_routes(r_0_0_L, r_13_0_R)
    assert [r.road_id for r in route_0_to_13[0].roads] == ["0_0_L", "15_0_R", "13_0_R"]
    assert (
        route_0_to_13[0].road_length
        == r_0_0_L.length + r_13_0_R.length + r_15_0_R.length
    )

    # distance between points along route
    start_point = Point(x=118.0, y=150.0, z=0.0)
    end_point = Point(x=190.0, y=118.0, z=0.0)
    assert round(route_0_to_13[0].distance_between(start_point, end_point), 2) == 95.93

    # project along route
    candidates = route_0_to_13[0].project_along(start_point, 100)
    assert len(candidates) == 4

    r_13_0_L = road_map.road_by_id("13_0_L")
    r_9_0_R = road_map.road_by_id("9_0_R")
    r_0_0_R = road_map.road_by_id("0_0_R")
    route_13_to_0 = road_map.generate_routes(r_13_0_L, r_0_0_R)
    assert [r.road_id for r in route_13_to_0[0].roads] == ["13_0_L", "9_0_R", "0_0_R"]
    assert (
        route_13_to_0[0].road_length
        == r_13_0_L.length + r_9_0_R.length + r_0_0_R.length
    )

    # distance between points along route
    start_point = Point(x=150.0, y=121.0, z=0.0)
    end_point = Point(x=122.0, y=180.0, z=0.0)
    assert round(route_13_to_0[0].distance_between(start_point, end_point), 2) == 84.44

    # project along route
    candidates = route_13_to_0[0].project_along(start_point, 100)
    assert len(candidates) == 4

    # Invalid route generation
    invalid_route = road_map.generate_routes(
        road_map.road_by_id("13_0_L"), road_map.road_by_id("1_0_L")
    )
    assert [r.road_id for r in invalid_route[0].roads] == []


def test_od_map_figure_eight():
    root = path.join(Path(__file__).parent.absolute(), "maps")
    road_map = OpenDriveRoadNetwork.from_file(path.join(root, "Figure-Eight.xodr"))
    assert isinstance(road_map, OpenDriveRoadNetwork)

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0

    # Road tests
    r_508_0_R = road_map.road_by_id("508_0_R")
    assert r_508_0_R
    assert not r_508_0_R.is_junction
    assert len(r_508_0_R.lanes) == 4
    assert set([r.road_id for r in r_508_0_R.incoming_roads]) == {"516_0_R"}
    assert set([r.road_id for r in r_508_0_R.outgoing_roads]) == {"501_0_L"}
    assert len(r_508_0_R.shape().exterior.coords) == 1603

    r_508_0_L = road_map.road_by_id("508_0_L")
    assert r_508_0_L
    assert not r_508_0_L.is_junction
    assert len(r_508_0_L.lanes) == 4
    assert set([r.road_id for r in r_508_0_L.incoming_roads]) == {"501_0_R"}
    assert set([r.road_id for r in r_508_0_L.outgoing_roads]) == {"516_0_L"}
    assert len(r_508_0_L.shape().exterior.coords) == 1603

    # Lane tests
    l1 = road_map.lane_by_id("508_0_R_-1")
    assert l1
    assert l1.road.road_id == "508_0_R"
    assert l1.index == -1
    assert l1.is_drivable
    assert len(l1.shape().exterior.coords) == 1603

    assert len(l1.lanes_in_same_direction) == 3
    assert round(l1.length, 2) == 541.50

    assert set([l.lane_id for l in l1.incoming_lanes]) == {"516_0_R_-1"}
    assert set([l.lane_id for l in l1.outgoing_lanes]) == {"501_0_L_1"}

    # point on straight part of the lane
    point = (13.0, -17.0, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 7.21
    assert round(refline_pt.t, 2) == -0.95
    central_point = l1.center_at_point(point)
    assert (round(central_point.x, 2), round(central_point.y, 2)) == (13.67, -16.33)

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.75
    assert l1.curvature_radius_at_offset(offset) == 1407374883553280.0
    assert l1.contains_point(point)
    assert l1.road.contains_point(point)

    # point on curved part of the lane
    point = (163.56, 75.84, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 364.39
    assert round(refline_pt.t, 2) == 0.13

    offset = refline_pt.s
    assert l1.width_at_offset(offset) == 3.75
    assert round(l1.curvature_radius_at_offset(offset), 2) == 81.87
    assert l1.contains_point(point)
    assert l1.road.contains_point(point)

    # oncoming lanes at this point
    assert set([l.lane_id for l in l1.oncoming_lanes_at_offset(offset)]) == {
        "508_0_L_1"
    }

    # edges on curved part
    left_edge, right_edge = l1.edges_at_point(point)
    assert (round(left_edge.x, 2), round(left_edge.y, 2)) == (162.63, 74.36)
    assert (round(right_edge.x, 2), round(right_edge.y, 2)) == (164.62, 77.53)

    # check for locations (lane, offset tuples) within distance at this offset
    candidates = l1.project_along(offset, 300)
    assert (len(candidates)) == 12

    # point not on lane but on road
    point = (163, 82, 0)
    refline_pt = l1.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 367.96
    assert round(refline_pt.t, 2) == -4.87
    assert not l1.contains_point(point)
    assert l1.road.contains_point(point)

    # nearest lanes
    point = (1.89, 0.79, 0)
    l5 = road_map.nearest_lane(point)
    assert l5.lane_id == "510_0_R_-1"
    assert l5.road.road_id == "510_0_R"
    assert l5.index == -1
    assert l5.road.contains_point(point)
    assert l5.is_drivable


def test_od_map_lane_offset():
    root = path.join(Path(__file__).parent.absolute(), "maps")
    file_path = path.join(root, "Ex_Simple-LaneOffset.xodr")
    road_map = OpenDriveRoadNetwork.from_file(file_path)
    assert isinstance(road_map, OpenDriveRoadNetwork)
    assert road_map.source == file_path
    assert road_map.bounding_box.max_pt == Point(x=100.0, y=8.0, z=0)
    assert road_map.bounding_box.min_pt == Point(x=0.0, y=-5.250000000000002, z=0)

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0

    # Nonexistent road/lane tests
    assert road_map.road_by_id("") is None
    assert road_map.lane_by_id("") is None

    # Surface tests
    surface = road_map.surface_by_id("1_1_R")
    assert surface.surface_id == "1_1_R"

    # Road tests
    r_1_0_R = road_map.road_by_id("1_0_R")
    assert r_1_0_R
    assert len(r_1_0_R.lanes) == 2
    assert not r_1_0_R.is_junction
    assert set([r.road_id for r in r_1_0_R.incoming_roads]) == set()
    assert set([r.road_id for r in r_1_0_R.outgoing_roads]) == {"1_1_R"}

    r_1_0_L = road_map.road_by_id("1_0_L")
    assert r_1_0_L
    assert len(r_1_0_L.lanes) == 3
    assert not r_1_0_L.is_junction
    assert set([r.road_id for r in r_1_0_L.incoming_roads]) == {"1_1_L"}
    assert set([r.road_id for r in r_1_0_L.outgoing_roads]) == set()

    r_1_1_R = road_map.road_by_id("1_1_R")
    assert r_1_1_R
    assert len(r_1_1_R.lanes) == 3
    assert not r_1_1_R.is_junction
    assert set([r.road_id for r in r_1_1_R.incoming_roads]) == {"1_0_R"}
    assert set([r.road_id for r in r_1_1_R.outgoing_roads]) == {"1_2_R"}
    assert set([s.surface_id for s in r_1_1_R.entry_surfaces]) == {"1_0_R"}
    assert set([s.surface_id for s in r_1_1_R.exit_surfaces]) == {"1_2_R"}

    r_1_1_L = road_map.road_by_id("1_1_L")
    assert r_1_1_L
    assert len(r_1_1_L.lanes) == 3
    assert not r_1_1_L.is_junction
    assert set([r.road_id for r in r_1_1_L.incoming_roads]) == {"1_2_L"}
    assert set([r.road_id for r in r_1_1_L.outgoing_roads]) == {"1_0_L"}
    assert set([s.surface_id for s in r_1_1_L.entry_surfaces]) == {"1_2_L"}
    assert set([s.surface_id for s in r_1_1_L.exit_surfaces]) == {"1_0_L"}

    r_1_2_R = road_map.road_by_id("1_2_R")
    assert r_1_2_R
    assert len(r_1_2_R.lanes) == 3
    assert not r_1_2_R.is_junction
    assert set([r.road_id for r in r_1_2_R.incoming_roads]) == {"1_1_R"}
    assert set([r.road_id for r in r_1_2_R.outgoing_roads]) == set()

    r_1_2_L = road_map.road_by_id("1_2_L")
    assert r_1_2_L
    assert len(r_1_2_L.lanes) == 2
    assert not r_1_2_L.is_junction
    assert set([r.road_id for r in r_1_2_L.incoming_roads]) == set()
    assert set([r.road_id for r in r_1_2_L.outgoing_roads]) == {"1_1_L"}

    # Lane tests
    l0 = road_map.lane_by_id("1_1_L_1")
    assert l0
    assert l0.road.road_id == "1_1_L"
    assert l0.index == 1
    assert l0.is_drivable

    assert set([lane.lane_id for lane in l0.incoming_lanes]) == set()
    assert set([lane.lane_id for lane in l0.outgoing_lanes]) == {"1_0_L_1"}
    assert set([lane.lane_id for lane in l0.entry_surfaces]) == set()
    assert set([lane.lane_id for lane in l0.exit_surfaces]) == {"1_0_L_1"}

    right_lane, direction = l0.lane_to_right
    assert right_lane
    assert direction
    assert right_lane.lane_id == "1_1_L_2"
    assert right_lane.index == 2

    left_lane, direction = l0.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "1_1_R_-1"
    assert left_lane.index == -1

    further_right_lane, direction = right_lane.lane_to_right
    assert further_right_lane
    assert direction
    assert further_right_lane.lane_id == "1_1_L_3"
    assert further_right_lane.index == 3

    l1 = road_map.lane_by_id("1_1_R_-1")
    assert l1
    assert l1.road.road_id == "1_1_R"
    assert l1.index == -1
    assert l1.is_drivable

    left_lane, direction = l1.lane_to_left
    assert left_lane
    assert not direction
    assert left_lane.lane_id == "1_1_L_1"
    assert left_lane.index == 1

    # point on lane
    point = (31.0, 2.0, 0)
    refline_pt = l0.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 43.52
    assert round(refline_pt.t, 2) == -0.31

    offset = refline_pt.s
    assert round(l0.width_at_offset(offset), 2) == 3.1
    assert round(l0.curvature_radius_at_offset(offset), 2) == -291.53
    assert l0.contains_point(point)
    assert l0.road.contains_point(point)

    # lane edges on point
    left_edge, right_edge = l0.edges_at_point(point)
    assert (round(left_edge.x, 2), round(left_edge.y, 2)) == (31.08, 0.13)
    assert (round(right_edge.x, 2), round(right_edge.y, 2)) == (31.0, 3.25)

    # road edges on point
    road_left_edge, road_right_edge = r_1_1_R.edges_at_point(point)
    assert (round(road_left_edge.x, 2), round(road_left_edge.y, 2)) == (31.08, 0.13)
    assert (round(road_right_edge.x, 2), round(road_right_edge.y, 2)) == (31.0, -5.25)

    # point not on lane but on road
    point = (31.0, 4.5, 0)
    refline_pt = l0.to_lane_coord(point)
    assert round(refline_pt.s, 2) == 43.47
    assert round(refline_pt.t, 2) == -2.81

    offset = refline_pt.s
    assert round(l0.width_at_offset(offset), 2) == 3.1
    assert round(l0.curvature_radius_at_offset(offset), 2) == -292.24
    assert not l0.contains_point(point)
    assert l0.road.contains_point(point)

    # check for locations (lane, offset tuples) within distance at this offset
    candidates = l0.project_along(offset, 20)
    assert (len(candidates)) == 3

    # nearest lanes for a point in lane
    point = (60.0, -2.38, 0)
    l4 = road_map.nearest_lane(point)
    assert l4.lane_id == "1_1_R_-2"
    assert l4.road.road_id == "1_1_R"
    assert l4.index == -2
    assert l4.road.contains_point(point)
    assert l4.is_drivable

    # get the road for point containing it
    point = (80.0, 1.3, 0)
    r4 = road_map.road_with_point(point)
    assert r4.road_id == "1_2_R"

    # route generation
    start = road_map.road_by_id("1_0_R")
    end = road_map.road_by_id("1_2_R")
    route = road_map.generate_routes(start, end)
    assert [r.road_id for r in route[0].roads] == ["1_0_R", "1_1_R", "1_2_R"]

    # distance between points along route
    start_point = Point(x=17.56, y=-1.67, z=0.0)
    end_point = Point(x=89.96, y=2.15, z=0.0)
    assert round(route[0].distance_between(start_point, end_point), 2) == 72.4
    # project along route
    candidates = route[0].project_along(start_point, 70)
    assert len(candidates) == 3


def test_od_map_motorway():
    root = path.join(Path(__file__).parent.absolute(), "maps")
    file_path = path.join(root, "UC_Motorway-Exit-Entry.xodr")
    road_map = OpenDriveRoadNetwork.from_file(file_path)
    assert isinstance(road_map, OpenDriveRoadNetwork)
    assert road_map.source == file_path

    # Expected properties for all roads and lanes
    for road_id, road in road_map._roads.items():
        assert type(road_id) == str
        assert road.is_junction is not None
        assert road.length is not None
        assert road.length >= 0
        assert road.parallel_roads == []
        for lane in road.lanes:
            assert lane.in_junction is not None
            assert lane.length is not None
            assert lane.length >= 0

    # route generation
    empty_route = road_map.empty_route()
    assert empty_route

    random_route = road_map.random_route(10)
    assert random_route.roads

    route_6_to_40 = road_map.generate_routes(
        road_map.road_by_id("6_0_L"), road_map.road_by_id("40_0_R")
    )
    assert [r.road_id for r in route_6_to_40[0].roads] == [
        "6_0_L",
        "18_1_L",
        "18_0_L",
        "28_0_R",
        "42_0_R",
        "43_0_R",
        "5_0_R",
        "5_1_R",
        "5_2_R",
        "8_0_R",
        "40_0_R",
    ]

    # distance between points along route
    start_point = Point(x=222.09, y=998.12, z=0.0)
    end_point = Point(x=492.62, y=428.18, z=0.0)
    assert round(route_6_to_40[0].distance_between(start_point, end_point), 2) == 761.66

    # project along route
    candidates = route_6_to_40[0].project_along(start_point, 600)
    assert len(candidates) == 6

    route_6_to_34_via_19 = road_map.generate_routes(
        road_map.road_by_id("6_0_L"),
        road_map.road_by_id("34_0_R"),
        [road_map.road_by_id("19_0_L"), road_map.road_by_id("17_0_R")],
    )
    assert [r.road_id for r in route_6_to_34_via_19[0].roads] == [
        "6_0_L",
        "18_1_L",
        "18_0_L",
        "11_0_R",
        "19_2_L",
        "19_1_L",
        "19_0_L",
        "27_0_R",
        "17_0_R",
        "12_0_R",
        "33_0_R",
        "33_1_R",
        "33_2_R",
        "39_0_R",
        "34_0_R",
    ]

    # distance between points along route
    start_point = Point(x=222.09, y=998.12, z=0.0)
    end_point = Point(x=507.40, y=1518.31, z=0.0)
    assert (
        round(route_6_to_34_via_19[0].distance_between(start_point, end_point), 2)
        == 971.71
    )
    # project along route
    candidates = route_6_to_34_via_19[0].project_along(start_point, 600)
    assert len(candidates) == 3

    route_34_to_6 = road_map.generate_routes(
        road_map.road_by_id("34_0_L"), road_map.road_by_id("6_0_R")
    )

    assert [r.road_id for r in route_34_to_6[0].roads] == [
        "34_0_L",
        "38_0_R",
        "36_1_L",
        "36_0_L",
        "4_0_R",
        "13_0_R",
        "21_0_R",
        "21_1_R",
        "35_0_R",
        "18_0_R",
        "18_1_R",
        "6_0_R",
    ]

    # distance between points along route
    start_point = Point(x=493.70, y=1528.79, z=0.0)
    end_point = Point(x=192.60, y=1001.47, z=0.0)
    assert round(route_34_to_6[0].distance_between(start_point, end_point), 2) == 1114.0
    # project along route
    candidates = route_34_to_6[0].project_along(start_point, 600)
    assert len(candidates) == 4


def lane_points(lane):
    polygon: Polygon = lane.shape()
    xs, ys = [], []
    for x, y in polygon.exterior.coords:
        xs.append(x)
        ys.append(y)
    return xs, ys


def lp_points(lps):
    xs, ys = [], []
    for lp in lps:
        xs.append(lp.lp.pose.position[0])
        ys.append(lp.lp.pose.position[1])
    return xs, ys


def visualize():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    root = path.join(Path(__file__).parent.absolute(), "maps")
    filename = "Ex_Simple-LaneOffset.xodr"
    filepath = path.join(root, filename)
    road_map = OpenDriveRoadNetwork.from_file(filepath, lanepoint_spacing=0.5)

    roads = road_map._roads
    lanepoint_by_lane_memo = {}
    shape_lanepoints = []

    for road_id in roads:
        road = roads[road_id]
        for lane in road.lanes:
            xs, ys = lane_points(lane)
            plt.plot(xs, ys, "k-")
            if lane.is_drivable:
                _, new_lps = road_map._lanepoints._shape_lanepoints_along_lane(
                    lane, lanepoint_by_lane_memo
                )
                shape_lanepoints += new_lps
                xs, ys = lp_points(new_lps)
                plt.scatter(xs, ys, s=1, c="r")

    ax.set_title(filename)
    ax.axis("equal")
    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


if __name__ == "__main__":
    visualize()
