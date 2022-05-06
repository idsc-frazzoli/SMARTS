from baselines.marl_benchmark.evaluation.utils import StraightLane, Map


def get_merge110_lanes2():
    # Map and Lanes for merge110_lanes2 scenario
    merge110_lanes2_lanes = [StraightLane(boundaries=[((0.0, 51.3), (10.0, 0.0)), ((-1.2, 33.5), (3.8, -3.3))],
                                          center_lines=[((-0.6, 32.5), (7.0, 0.4))]
                                          ),
                             StraightLane(boundaries=[((0.0, 33.5), (-10.0, -3.3)), ((1.2, 51.3), (-16.3, -6.2))],
                                          center_lines=[((0.6, 32.5), (-13.2, -6.7))]
                                          ),
                             StraightLane(boundaries=[((51.3, 110.0), (0.0, 0.0)), ((51.3, 110.0), (-6.2, -6.2))],
                                          center_lines=[((51.3, 110.0), (-3.1, -3.1))]
                                          ),
                             ]
    merge110_lanes2_map = Map(lanes=merge110_lanes2_lanes,
                              x_lim=(-5.0, 115.0), y_lim=(-17.0, 11.0),
                              aspect_ratio=(15, 4))

    return merge110_lanes2_map


def get_merge40_lanes1():
    # Map and Lanes for merge40_lanes1 scenario
    merge40_lanes1_lanes = [StraightLane(boundaries=[((0.0, 15.4), (10.9, 0.0)), ((-1.8, 12.4), (8.3, -1.6))],
                                         center_lines=[]
                                         ),
                            StraightLane(boundaries=[((-3.1, 12.4), (-12.5, -1.6)), ((-1.4, 16.3), (-15.1, -3.1))],
                                         center_lines=[]
                                         ),
                            StraightLane(boundaries=[((15.4, 40.6), (0.0, 0.0)), ((16.3, 40.6), (-3.1, -3.1))],
                                         center_lines=[]
                                         ),
                            ]
    merge40_lanes1_map = Map(lanes=merge40_lanes1_lanes,
                             x_lim=(-7.0, 43.0), y_lim=(-18.0, 13.0),
                             aspect_ratio=(12, 7))

    return merge40_lanes1_map


def get_merge75_lanes321():
    # Map for merge75_lanes321 scenario
    merge75_lanes321_lanes = []
    merge75_lanes321_map = Map(lanes=merge75_lanes321_lanes,
                               x_lim=(-5.0, 80.0), y_lim=(-11.0, 1.0),
                               aspect_ratio=(17, 3))

    return merge75_lanes321_map


def get_merge90_lanes32():
    merge90_lanes32_lanes = [StraightLane(boundaries=[((0, 43.4), (-3.2, -3.2)), ((0, 43.4), (0, 0))],
                                          center_lines=[]),
                             StraightLane(boundaries=[((0, 31.3), (-9.4, -9.4)), ((0, 31.3), (-6.3, -6.3))],
                                          center_lines=[]),
                             StraightLane(boundaries=[((0, 31.3), (3.1, 3.1)), ((0, 31.3), (6.2, 6.2))],
                                          center_lines=[]),
                             StraightLane(boundaries=[((31.3, 43.4), (3.1, 0)), ((31.3, 55.4), (6.2, 0))],
                                          center_lines=[]),
                             StraightLane(boundaries=[((31.3, 43.4), (-6.3, -3.2)), ((31.3, 45), (-9.4, -6.4))],
                                          center_lines=[]),
                             StraightLane(boundaries=[((45, 90.6), (-6.3, -6.3)), ((55.4, 90.6), (0, 0))],
                                          center_lines=[((55.4, 90), (-3.15, -3.15))])]
    merge90_lanes32_map = Map(lanes=merge90_lanes32_lanes,
                              x_lim=(-2.0, 92.0), y_lim=(-10.5, 7.5),
                              aspect_ratio=(17, 4))

    return merge90_lanes32_map


def get_merge65_lanes42():
    merge65_lanes42_lanes = [StraightLane(boundaries=[((0, 18.6), (12.4, 12.4)), ((0, 17.9), (6.1, 6.1))],
                                          center_lines=[((0, 18.3), (9.25, 9.25))]),  # E0
                             StraightLane(boundaries=[((17.9, 28.8), (6.1, 3)), ((18.6, 40.6), (12.4, 6.1))],
                                          center_lines=[((18.3, 40.7), (9.25, 3.05))]),  # E4
                             StraightLane(boundaries=[((40.6, 65.6), (6.1, 6.1)), ((40.6, 65.6), (0, 0))],
                                          center_lines=[((40.7, 65.6), (3.05, 3.05))]),  # E6
                             StraightLane(boundaries=[((0, 18.7), (0, 0)), ((0, 19.3), (-6.4, -6.4))],
                                          center_lines=[((0, 19), (-3.2, -3.2))]),  # E3
                             StraightLane(boundaries=[((18.7, 28.8), (0, 3)), ((19.3, 40.6), (-6.4, 0))],
                                          center_lines=[((19, 40.7), (-3.2, 3.05))]),  # E5
                             ]
    merge65_lanes42_map = Map(lanes=merge65_lanes42_lanes,
                              x_lim=(-2.0, 67.0), y_lim=(-8, 14),
                              aspect_ratio=(16, 5))

    return merge65_lanes42_map


def get_empty():
    return Map(None, None, None, aspect_ratio=(12, 6), empty=True)

