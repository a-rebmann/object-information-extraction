
class Event(object):

    def __init__(self, label, timestamp, vmap):
        self.label = label
        self.timestamp = timestamp
        self.vmap = vmap
        self.omap = set()
        self.object_type_to_actions = dict()
        self.is_duplicate = False

    def __repr__(self):
        return f'Event(label={self.label}, time={self.timestamp}, vmap={self.vmap}, omap={self.omap}, obj_type_to_act={self.object_type_to_actions})'
