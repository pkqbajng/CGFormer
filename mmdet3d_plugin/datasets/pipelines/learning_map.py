learning_map = {
    'kitti': {
          0 : 0,     # "unlabeled"
          1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
          10: 1,     # "car"
          11: 2,     # "bicycle"
          13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
          15: 3,     # "motorcycle"
          16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
          18: 4,     # "truck"
          20: 5,     # "other-vehicle"
          30: 6,     # "person"
          31: 7,     # "bicyclist"
          32: 8,     # "motorcyclist"
          40: 9,     # "road"
          44: 10,    # "parking"
          48: 11,    # "sidewalk"
          49: 12,    # "other-ground"
          50: 13,    # "building"
          51: 14,    # "fence"
          52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
          60: 9,     # "lane-marking" to "road" ---------------------------------mapped
          70: 15,    # "vegetation"
          71: 16,    # "trunk"
          72: 17,    # "terrain"
          80: 18,    # "pole"
          81: 19,    # "traffic-sign"
          99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
          252: 1,    # "moving-car" to "car" ------------------------------------mapped
          253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
          254: 6,    # "moving-person" to "person" ------------------------------mapped
          255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
          256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
          257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
          258: 4,    # "moving-truck" to "truck" --------------------------------mapped
          259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
    }
}