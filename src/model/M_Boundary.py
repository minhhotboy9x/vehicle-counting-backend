import supervision as sv
from supervision.geometry.core import Point, Position
from supervision.draw.color import Color
from pymongo import UpdateOne
from model.utils import ccw
class Boundary:
    mongo = None
    line_counters = []
    line_annotators = []

    @classmethod
    def init_mongo(cls, mongo):
        cls.mongo = mongo
    
    def __init__(self, id, camId, pointL, pointR, pointDirect):
        self.id = id
        self.camId = camId
        self.pointL = pointL
        self.pointR = pointR
        self.pointDirect = pointDirect
    

    def json(self):
        return {
            'id': self.id,
            'camId': self.camId,
            'pointL': self.pointL,
            'pointR': self.pointR,
            'pointDirect': self.pointDirect
        }

    @classmethod
    def insert(self, data):
        return Boundary.mongo.db.boundary.insert_one(data)
    
    @classmethod
    def update(cls, id, **kwargs):
        query = {'id': id}
        new_values = {'$set': kwargs}
        result = cls.mongo.db.boundary.update_one(query, new_values)
        return result.acknowledged
    
    @classmethod
    def update_or_insert(cls, id, **kwargs):
        # Tìm kiếm bản ghi có id tương ứng trong cơ sở dữ liệu
        query = {'id': id}
        existing_data = cls.mongo.db.boundary.find_one(query)
        if existing_data:
            # Nếu tồn tại bản ghi, thực hiện cập nhật
            new_values = {'$set': kwargs}
            result = cls.mongo.db.boundary.update_one(query, new_values)
            return result.acknowledged
        else:
            # Nếu không tồn tại bản ghi, thực hiện insert
            new_data = {'id': id,
                        'in': 0,
                        'out': 0,
                         **kwargs}
            result = cls.mongo.db.boundary.insert_one(new_data)
            return result.acknowledged
    
    @classmethod
    def update_many(cls, update_operations: list[UpdateOne]):
        if update_operations == []:
            return
        cls.mongo.db.boundary.bulk_write(update_operations)

    @classmethod
    def delete(cls, query):
        return cls.mongo.db.boundary.delete_one(query)
    
    @classmethod
    def find(cls, query):
        return list(cls.mongo.db.boundary.find(query))
    
    @classmethod
    def get_line_annotators(cls, boundaries_list):
        cls.line_counters = []
        cls.line_annotators = []
        for boundary in boundaries_list:
            point1 = Point(boundary['pointL']['x'] + 7.5, boundary['pointL']['y'] + 7.5)
            point2 = Point(boundary['pointR']['x'] + 7.5, boundary['pointR']['y'] + 7.5)
            if ccw(point1, point2) > 0:
                point1, point2 = point2, point1
            linezone = sv.LineZone(point1, point2,
                                [Position.BOTTOM_CENTER, Position.TOP_CENTER])
            # Thêm linezone đã chỉnh sửa vào danh sách line_counters
            linezone.id = boundary['id']
            linezone.in_count = boundary.get('in') if boundary.get('in') is not None else 0
            linezone.out_count = boundary.get('out') if boundary.get('out') is not None else 0
            cls.line_counters.append(linezone)
            cls.line_annotators.append(sv.LineZoneAnnotator(thickness = 1, text_thickness=1, text_padding=5))
        
