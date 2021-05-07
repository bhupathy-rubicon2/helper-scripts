import json
import os

CAMERA_SPECS = {
    "onePlus5":
        {
            "focalLength": 4.103,
            "sensorHeight": 5.22,
            "sensorWidth": 2.92
        },
    "piCamera2":
        {
            "focalLength": 3.04,
            "sensorHeight": 2.76,
            "sensorWidth": 3.68
        },

    "goPro5":
        {
            "focalLength": 5.64,
            "sensorHeight": 4.5,
            "sensorWidth": 6.2
        },
    "ELP_OV2710":
        {

            "focalLength": 3.6,
            "sensorHeight": 5.856,
            "sensorWidth": 3.276
        }
}


class DistanceModel(object):
    """
        Provides a simple method for estimating distances based on real world object dimensions & camera lens properties
    """

    def __init__(self, device, dimensions_json):
        """

        :param device: string that specifies which camera specifications to use. Must be configured in CAMERA_SPECS dict.
        :param dimensions_json: file containing real world dimensions
        """
        self.device = device
        if not os.path.exists(dimensions_json):
            raise ValueError("%s does not exits" % dimensions_json)
        with open(dimensions_json) as f:
            self.dimensions = json.load(f)
        # Device specific camera properties in mm
        if self.device in CAMERA_SPECS:
            device_spec = CAMERA_SPECS[self.device]
            self.focal_length = device_spec['focalLength']
            self.sensor_height = device_spec['sensorHeight']
            self.sensor_width = device_spec['sensorWidth']
        else:
            raise NotImplementedError("%s not in list supported devices: %s"
                                      % (self.device, ','.join(CAMERA_SPECS.keys()))
                                      )

    def __get_dimension_by_class(self, cls, dimension):
        return self.dimensions[cls.capitalize()][dimension]

    def distance_to_obj_by_h(self, detection_height, img_height, cls):
        h = self.__get_dimension_by_class(cls, dimension="height")
        return (self.focal_length * h * img_height) / (detection_height * self.sensor_height) / 1000

    def distance_to_obj_by_w(self, detection_width, img_width, cls):
        w = self.__get_dimension_by_class(cls, dimension="width")
        return (self.focal_length * w * img_width) / (detection_width * self.sensor_width) / 1000
