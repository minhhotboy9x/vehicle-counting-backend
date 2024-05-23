import time

class Violation:
    mongo = None

    @classmethod
    def init_mongo(cls, mongo):
        cls.mongo = mongo