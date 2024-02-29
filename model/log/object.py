
class ObjectInstance:

    def __init__(self, oid, obj_type):
        self.oid = oid
        self.obj_type = obj_type
        self.vmap = dict()
        self.discovered_from = set()

