import supervision as sv
import numpy as np
from config import FRAME_WIDTH, FRAME_HEIGHT
from supervision.draw.color import Color
class Roi:
    mongo = None
    polygon_counters = []
    polygon_annotators = []

    @classmethod
    def init_mongo(cls, mongo):
        cls.mongo = mongo
    
    def __init__(self, id, camId, points):
        self.id = id
        self.camId = camId
        self.points = points
    
    @classmethod
    def insert(self, data):
        return Roi.mongo.db.roi.insert_one(data)
    
    @classmethod
    def update(cls, id, **kwargs):
        query = {'id': id}
        new_values = {'$set': kwargs}
        result = cls.mongo.db.roi.update_one(query, new_values)
        return result.acknowledged
    
    @classmethod
    def update_or_insert(cls, id, **kwargs):
        # Tìm kiếm bản ghi có id tương ứng trong cơ sở dữ liệu
        query = {'id': id}
        existing_data = cls.mongo.db.roi.find_one(query)
        if existing_data:
            # Nếu tồn tại bản ghi, thực hiện cập nhật
            new_values = {'$set': kwargs}
            result = cls.mongo.db.roi.update_one(query, new_values)
            return result.acknowledged
        else:
            # Nếu không tồn tại bản ghi, thực hiện insert
            new_data = {'id': id,
                        'mapping points': [
                                {"x": 100, "y": 100},
                                {"x": 200, "y": 100},
                                {"x": 200, "y": 200},
                                {"x": 100, "y": 200}
                            ],
                         **kwargs}
            result = cls.mongo.db.roi.insert_one(new_data)
            return result.acknowledged
        
    @classmethod
    def delete(cls, query):
        return cls.mongo.db.roi.delete_one(query)
    
    @classmethod
    def find(cls, query):
        return list(cls.mongo.db.roi.find(query))

    @classmethod
    def get_polygon_annotators(cls, roi_list):
        points = [roi['points'] for roi in roi_list]
        points = np.array([[[coord['x'] + 7, coord['y'] + 7] for coord in polygon] for polygon in points])
        cls.polygon_counters = [sv.PolygonZone(point, [FRAME_WIDTH, FRAME_HEIGHT]) for point in points]
        cls.polygon_annotators = [sv.PolygonZoneAnnotator(counter, color=Color.RED, thickness=2, text_padding=5) for counter in cls.polygon_counters]