from const import CREATE_TERMS, CREATION, LIFECYCLE, CASE_NOTION, N_TO_ONE, N_TO_M, ONE_TO_N
from model.log.log import Log
from model.log.object import ObjectInstance
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from preprocessing.preprocessor import clean_attribute_name


class InstanceDiscoverer:

    def __init__(self, config):
        self.config = config
        self.object_to_id = {}
        self.property_candidates = {}
        self.obj_type_to_life_cycles = {}
        self.obj_type_to_start_events = {}
        self.obj_type_to_end_events = {}
        self.obj_type_to_mandatory_events = {}
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.cardinality = N_TO_ONE

    def is_covered(self, obj_type):
        return obj_type in self.object_to_id.keys()

    def check_for_creation_action(self, action):
        return (not isinstance(action, float)) and (self.lemmatizer.lemmatize(action) in CREATE_TERMS)

    def discover_instances(self, log: Log, case_obj_types_to_consider=None):
        self.extract_lifecycles(log)
        self.discover_explicit_instances(log)
        self.discover_implicit_instances(log)
        # here we can reassess the list of object types, i.e, if no instance has been found for an object type there is
        # no value of considering this object type
        self.reassess_obj_types(log)

    def discover_explicit_instances(self, log: Log):
        obj_type_to_id_att = {}
        taken_obj = set()
        #  case ID
        for obj_type in log.object_types:
            if all(obj_type in str(cid) or obj_type in str(cid).lower() for cid in
                   log.aug_log.df[log.case_id].unique()):
                obj_type_to_id_att[obj_type] = log.aug_log.case_id
                log.case_object = obj_type
        #  direct name matches
        for obj_type, att in log.object_from_case.items():
            if not log.isstring(att) and not log.isint(att):
                if obj_type not in self.property_candidates:
                    self.property_candidates[obj_type] = set()
                self.property_candidates[obj_type].add(att)
                continue
            obj_type_to_id_att[obj_type] = att
            print("Found name match for ID!", att, obj_type)

        # candidate match computation
        for att in log.selected_attributes:
            for obj_type in log.object_types:
                # name overlap
                if obj_type in clean_attribute_name(att) or clean_attribute_name(att) in obj_type:
                    print(att, log.check_uniqueness_across_cases(att, obj_ty=obj_type),
                          log.isstring(att) and log.aug_log.num_uniques[att] > 2)
                    if not log.isstring(att) and not log.isint(att):
                        if obj_type not in self.property_candidates:
                            self.property_candidates[obj_type] = set()
                        self.property_candidates[obj_type].add(att)
                        continue
                    if log.check_uniqueness_across_cases(att, obj_ty=obj_type) >= .9:
                        if obj_type in taken_obj or obj_type in log.object_from_case.keys():
                            continue
                        print("Found name match for ID!", att, obj_type)
                        obj_type_to_id_att[obj_type] = att
                    else:
                        if obj_type not in self.property_candidates:
                            self.property_candidates[obj_type] = set()
                        self.property_candidates[obj_type].add(att)

                # co-occurrence check
                elif obj_type in log.obj_types_from_event_atts and all(att in e.vmap.keys() for e in [lc[0] for lc in
                                                                                                      self.obj_type_to_life_cycles[
                                                                                                          obj_type] if
                                                                                                      len(lc) > 0]) and all(
                    att not in e.vmap.keys() for e in log.events if
                    obj_type not in e.object_type_to_actions.keys()):
                    print(att, log.check_uniqueness_across_cases(att, obj_ty=obj_type),
                          log.isstring(att) and log.aug_log.num_uniques[att] > 2)
                    if not log.isstring(att) and not log.isint(att):
                        if obj_type not in self.property_candidates:
                            self.property_candidates[obj_type] = set()
                        if any(att not in e.vmap.keys() for e in log.events):
                            self.property_candidates[obj_type].add(att)
                        continue
                    if log.check_uniqueness_across_cases(att, obj_ty=obj_type) >= .9:
                        print("Found co-occurrence match for ID!", att, obj_type)
                        obj_type_to_id_att[obj_type] = att
                    else:
                        if obj_type not in self.property_candidates:
                            self.property_candidates[obj_type] = set()
                        if any(att not in e.vmap.keys() for e in log.events):
                            self.property_candidates[obj_type].add(att)

                elif (obj_type not in log.obj_types_from_event_atts or all(att in e.vmap.keys() for e in
                                                                           [lc[0] for lc in
                                                                            self.obj_type_to_life_cycles[obj_type] if
                                                                            len(lc) > 0])) and (
                        att in log.att_names_no_case_atts or log.case_object == obj_type):
                    if obj_type not in self.property_candidates:
                        self.property_candidates[obj_type] = set()
                    if any(att not in e.vmap.keys() for e in log.events):
                        self.property_candidates[obj_type].add(att)
        self.object_to_id = obj_type_to_id_att

    def get_life_cycles_from_flat_log(self, log, obj_type):
        return set(
            [tuple([e for e in case.events if obj_type in e.object_type_to_actions]) for _, case in log.cases.items()])

    def discover_implicit_instances(self, log):
        # Discovery based on creation
        self.discover_creation_of_instances(log)
        # Discovery based on lifecycle information
        self.discover_instances_from_lifecycle(log)

        for obj_type, att in self.object_to_id.items():
            print(obj_type, att, "!!!")
            for event in log.events:
                if att in event.vmap:
                    event.omap.add(event.vmap[att])
                    log.type_mapping[event.vmap[att]] = obj_type
                    if event.vmap[att] not in log.objects:
                        log.objects[event.vmap[att]] = ObjectInstance(oid=event.vmap[att], obj_type=obj_type)
                    log.objects[event.vmap[att]].discovered_from.add(event)
                    # remove the attribute from the event's vmap since it is now an instance id
                    del event.vmap[att]
        # Discovery of case object
        self.discover_case_object(log)

    def discover_creation_of_instances(self, log):
        for obj_type in log.object_types:
            if self.is_covered(obj_type):
                print("ID", obj_type)
                continue
            cnt = 0
            for event in log.events:
                if obj_type in event.object_type_to_actions and len(event.object_type_to_actions[obj_type]) > 0 and any(
                        self.check_for_creation_action(action) for action in event.object_type_to_actions[obj_type]):
                    if not event.is_duplicate:
                        cnt += 1
                        obj_id = str(event.timestamp) + "#" + obj_type + "#" + str(cnt)  # dup_id+
                        if not log.check_duplicate:
                            obj_id = obj_id + "#" + str(event.vmap[log.case_id])
                        if "19" in log.name and obj_type == "purchase order item":
                            obj_id = "poi#" + str(cnt)
                        event.omap.add(obj_id)
                        log.type_mapping[obj_id] = obj_type
                        log.objects[obj_id] = ObjectInstance(oid=obj_id, obj_type=obj_type)
                        log.objects[obj_id].discovered_from.add(event)
                        others = log.find_duplicates(event)
                        for other in others:
                            #other.omap.add(obj_id)
                            log.objects[obj_id].discovered_from.add(other)
                        self.object_to_id[obj_type] = CREATION

    def discover_instances_from_lifecycle(self, log):
        for obj_type in log.object_types:
            # only consider object types from event attribute values AND object types from attribute names, which are not case attributes
            if obj_type in log.obj_types_from_event_atts or (
                    obj_type in log.aug_log.case_bo_to_case_att and log.aug_log.case_bo_to_case_att[
                obj_type] not in log.aug_log.case_attributes and obj_type in self.object_to_id):
                life_cycles = self.obj_type_to_life_cycles[obj_type]
                start_events = set([life_cycle[0].label for life_cycle in life_cycles if len(life_cycle) > 0])
                end_events = set([life_cycle[-1].label for life_cycle in life_cycles if len(life_cycle) > 0])
                mandatory_events = set.intersection(
                    *[set([e.label for e in life_cycle]) for life_cycle in life_cycles if len(life_cycle) > 0])

                self.obj_type_to_start_events[obj_type] = start_events
                self.obj_type_to_end_events[obj_type] = end_events
                self.obj_type_to_mandatory_events[obj_type] = mandatory_events

                print(obj_type, end_events, "end events")
                print(obj_type, mandatory_events, "mandatory events")
                print(obj_type, start_events, "start events")
                if self.is_covered(obj_type):
                    print(CREATION, obj_type)
                    continue
                # print("mandatory", mandatory_events)
                # start_dict = {}
                # for life_cycle in life_cycles:
                #     if life_cycle[0].label not in start_dict:
                #         start_dict[life_cycle[0].label] = set()
                #     start_dict[life_cycle[0].label].add([e.label for e in life_cycle])
                cnt = 0
                for case_id, case in log.cases.items():
                    for event in case.events:
                        if event.label in start_events and event.label in mandatory_events:
                            if not event.is_duplicate:
                                cnt += 1
                                # dup_id = ""
                                # if obj_type in log.duplication_atts_per_obj:
                                #     dup_id = str([event.vmap[att] for att in log.duplication_atts_per_obj[obj_type] if
                                #                   att in event.vmap])
                                obj_id = str(event.timestamp) + "#" + obj_type + "#" + str(cnt)  # dup_id
                                if not log.check_duplicate:
                                    obj_id = obj_id + "#" + case_id
                                if "19" in log.name and obj_type == "purchase order item":
                                    obj_id = "poi#" + str(cnt)
                                event.omap.add(obj_id)
                                log.type_mapping[obj_id] = obj_type
                                log.objects[obj_id] = ObjectInstance(oid=obj_id, obj_type=obj_type)
                                log.objects[obj_id].discovered_from.add(event)
                                others = log.find_duplicates(event)
                                for other in others:
                                    #other.omap.add(obj_id)
                                    log.objects[obj_id].discovered_from.add(other)
                self.object_to_id[obj_type] = LIFECYCLE
                print(cnt, obj_type, "lifecycle")
            else:
                print(obj_type, "obj type from case attribute!")
                continue

    def discover_case_object(self, log):
        # look at convergence and divergence
        candidates = set()
        if log.case_id in self.object_to_id.values():
            # print(self.object_to_id[obj], "is the case notion")
            for _, case in log.cases.items():
                log.objects[case.id] = ObjectInstance(oid=case.id, obj_type=log.case_object)
                log.objects[case.id].discovered_from.update(log.cases[case.id].events)
                self.object_to_id[log.case_object] = CASE_NOTION
                self.fill_property_candidates(log)
            return
        for obj_type in log.object_types:
            n_to_one = False
            one_to_n = False
            n_to_m = False
            if obj_type not in log.obj_types_from_event_atts and log.case_id == "case:concept:name":
                continue
            if any(obj_type in e.object_type_to_actions for e in log.events if e.is_duplicate):
                n_to_one = True
            if any(
                    len(set([inst for e in case.events for inst in e.omap if log.type_mapping[inst] == obj_type])) > 1
                    for case in log.cases.values()):
                one_to_n = True
            if n_to_one and one_to_n:
                n_to_m = True
            if n_to_m:
                self.cardinality = N_TO_M
            elif one_to_n and not self.cardinality == N_TO_M:
                self.cardinality = ONE_TO_N
            if n_to_one or one_to_n:
                print(obj_type, "is not the case notion", "duplicates")
                pass
            elif len(log.objects_per_type[obj_type]) < len(log.cases) * 0.99:
                print(obj_type, "is not the case notion")
                pass
            else:
                print(obj_type, "Could be the case notion")
                candidates.add(obj_type)
        if len(candidates) == 1:
            log.case_object = candidates.pop()
            print(log.case_object, "set as case notion.")
            if self.is_covered(log.case_object):
                print(CASE_NOTION, log.case_object)
                self.fill_property_candidates(log)
                # for obj_i in log.objects_per_type[log.case_object]:

                return
            if self.object_to_id[log.case_object] in [LIFECYCLE, CREATION]:
                self.object_to_id[log.case_object] = log.case_id
                for _, case in log.cases.items():
                    log.objects[case.id] = ObjectInstance(oid=case.id, obj_type=log.case_object)
                    log.objects[case.id].discovered_from.update(log.cases[case.id].events)
                    self.object_to_id[log.case_object] = CASE_NOTION
                    self.fill_property_candidates(log)
        else:
            print("Try fallback based on log-level attribute values", log.cleaned_name)
            for obj_type in log.object_types:
                if obj_type in log.cleaned_name:
                    log.case_object = obj_type
                    if self.is_covered(log.case_object) and self.object_to_id[obj_type] in [LIFECYCLE, CREATION]:
                        print(log.case_object, "set as case notion.")
                        print(CASE_NOTION, log.case_object)
                        self.fill_property_candidates(log)
                        self.object_to_id[log.case_object] = log.case_id

                        for _, case in log.cases.items():
                            log.objects[case.id] = ObjectInstance(oid=case.id, obj_type=log.case_object)
                            log.objects[case.id].discovered_from.update(log.cases[case.id].events)
                            self.object_to_id[log.case_object] = CASE_NOTION
                            self.fill_property_candidates(log)
                        return

            print("Try fall back based on first objects in all cases")
            second_round_candidates = []
            for obj_type in log.obj_types_from_event_atts:
                if all(obj_type in case.events[0].object_type_to_actions for case_id, case in log.cases.items() if
                       obj_type in candidates):
                    second_round_candidates.append(obj_type)
            if len(second_round_candidates) == 1:
                log.case_object = second_round_candidates.pop()
                if self.is_covered(log.case_object):
                    print(log.case_object, "set as case notion.")
                    print(CASE_NOTION, log.case_object)
                    self.fill_property_candidates(log)
                    self.object_to_id[log.case_object] = log.case_id
                    for _, case in log.cases.items():
                        log.objects[case.id] = ObjectInstance(oid=case.id, obj_type=log.case_object)
                        log.objects[case.id].discovered_from.update(log.cases[case.id].events)
                        self.object_to_id[log.case_object] = CASE_NOTION
                        self.fill_property_candidates(log)
                    return

    def reassess_obj_types(self, log):
        to_del = set()
        for obj_type, objs in log.objects_per_type.items():
            if len(objs) == 0:
                log.object_types.remove(obj_type)
                to_del.add(obj_type)
            else:
                print(obj_type, len(objs))
        for obj_type in to_del:
            del log.objs_per_type[obj_type]
            if obj_type in self.property_candidates:
                del self.property_candidates[obj_type]
            for event in log.events:
                if obj_type in event.object_type_to_actions:
                    del event.object_type_to_actions[obj_type]
            print("removing", obj_type, "from object types due to lack of instances.")
        print(log.object_types)
        print(log.objects_per_type.keys())

    def extract_lifecycles(self, log):
        for obj_type in log.object_types:
            if obj_type in log.obj_types_from_event_atts or (
                    obj_type in log.aug_log.case_bo_to_case_att and log.aug_log.case_bo_to_case_att[
                obj_type] not in log.aug_log.case_attributes):
                life_cycles = self.get_life_cycles_from_flat_log(log=log, obj_type=obj_type)
                self.obj_type_to_life_cycles[obj_type] = life_cycles

    def fill_property_candidates(self, log):
        for att in log.selected_attributes:
            if log.case_object not in self.property_candidates:
                self.property_candidates[log.case_object] = set()
            # if any(att not in e.vmap.keys() for e in log.events):
            self.property_candidates[log.case_object].add(att)
