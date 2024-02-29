from const import TERMS_FOR_MISSING, XES_TIME, XES_NAME


class Case(object):

    def __init__(self, case_id, events):
        self.id = case_id
        self.events = events
        self._trace = None
        self._non_redundant_parts = []
        self._subcases = None
        self._unique_classes = None

    @property
    def trace(self):
        if self._trace is None:
            self._trace = [str(event.label) for event in self.events]
        return self._trace

    @property
    def variant_str(self):
        return ",".join(self.trace)

    @property
    def unique_event_classes(self):
        if self._unique_classes is None:
            self._unique_classes = list(set([event.label for event in self.events]))
        return self._unique_classes

    @property
    def subcases(self):
        if self._subcases is None:
            self._subcases = []
            for i, part in enumerate(self._non_redundant_parts):
                self._subcases.append(Case(case_id=self.id+'_'+str(i), events=self.events[part[0]:part[1]+1]))
        else:
            return [self]
        return self._subcases

    def att_trace(self, att):
        return [e.attributes[att] for e in self.events]

    def __repr__(self):
        return f'Case(id={self.id}, events={[e for e in self.events]})'

