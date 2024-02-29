from instancediscovery.instancediscoverer import InstanceDiscoverer
from model.log.log import Log
from const import STRICT_ORD, REVERSE_STRICT_ORD, ONE_TO_N, N_TO_ONE, N_TO_M


class InstanceAssigner:

    def __init__(self, config, log,  instance_discoverer: InstanceDiscoverer):
        self.config = config
        self.log = log
        self.instance_discoverer = instance_discoverer

    def get_object_level_behavioral_regularities(self, log: Log):
        # exclude object types stemming from case attributes, as these are of course always present in a case
        obj_types_to_consider = [ty for ty in log.object_types if
                                 ty in log.obj_types_from_event_atts or self.log.aug_log.case_bo_to_case_att[
                                     ty] not in self.log.aug_log.case_attributes]
        print(obj_types_to_consider)
        first_obj_type_occurrence_per_case = {}
        for _, case in log.cases.items():
            first_obj_type_occurrence = {obj_type: -1 for obj_type in obj_types_to_consider}
            for obj_type in obj_types_to_consider:
                # look for occurrence of the current object type
                for idx, event in enumerate(case.events):
                    if obj_type in event.object_type_to_actions.keys():
                        first_obj_type_occurrence[obj_type] = idx
                        # if we found an occurrence, we can move on to the next object type
                        break
            first_obj_type_occurrence_per_case[case.id] = first_obj_type_occurrence
        seen = set()
        regularities = set()
        for obj_type_right in obj_types_to_consider:
            for obj_type_left in obj_types_to_consider:
                if obj_type_right != obj_type_left and (obj_type_right, obj_type_left) not in seen:
                    if all(first_obj_type_occurrence[obj_type_left] < first_obj_type_occurrence[obj_type_right] if
                           first_obj_type_occurrence[obj_type_right] != -1 and first_obj_type_occurrence[
                               obj_type_right] != -1 else True for case_id, first_obj_type_occurrence in
                           first_obj_type_occurrence_per_case.items()):
                        # print("Behavioral regularity between", obj_type_left, obj_type_right, "detected!")
                        regularities.add((obj_type_left, obj_type_right, STRICT_ORD))
                        print(obj_type_left, obj_type_right, STRICT_ORD)
                        regularities.add((obj_type_right, obj_type_left, REVERSE_STRICT_ORD))
                        seen.add((obj_type_right, obj_type_left))
                        seen.add((obj_type_left, obj_type_right))
        return regularities


    def assignment_general(self, log: Log, object_to_property):
        # If an instance was discovered from the specific event e, it is already there.
        # An object instance was discovered from an event f that happened before and is part of the same case c
        # AND the lifecycle of an object instance o in c has completed
        # AND no instance of type(o) was discovered in e.

        for _, case in log.cases.items():
            case_instances = set()

            previous_events_without_inst = {obj_type: [] for obj_type in log.objects_per_type.keys()}

            curr_inst_per_type = {obj_type: None for obj_type in log.objects_per_type.keys()}
            case_object_instances = set()

            for event in case.events:
                """
                1:1, 1:N
                we first assign the closest active object instance to events that have the type of that object instance
                but no other instance of that type already
                """
                for obj_type in event.object_type_to_actions.keys():
                    for att, val in event.vmap.items():
                        if log.isstring(att) and val in log.objects:
                            event.omap.add(val)
                    if obj_type not in log.obj_types_from_event_atts:
                        continue
                    if event.label in self.instance_discoverer.obj_type_to_start_events[obj_type]:
                        all_inst = [oi for oi in event.omap if log.type_mapping[oi] == obj_type]

                        if len(all_inst) > 0:
                            curr_inst_per_type[obj_type] = all_inst[0]

                            # Here we handle events that occured previously in the case,
                            # contain a object type for which they do not have an instance.
                            new_prev_list = []
                            for ev in previous_events_without_inst[obj_type]:
                                if all(att in ev.vmap and ev.vmap[att] == val for att, val in
                                       log.objects[curr_inst_per_type[obj_type]].vmap.items() if
                                       att in object_to_property[obj_type]):
                                    ev.omap.add(curr_inst_per_type[obj_type])
                                else:
                                    new_prev_list.append(ev)
                            previous_events_without_inst[obj_type] = new_prev_list
                        else:
                            curr_inst_per_type[obj_type] = None
                    if not any(log.type_mapping[oi] == obj_type for oi in event.omap):
                        if curr_inst_per_type[obj_type]:
                            # check the object properties, if the values differ, it cannot be the same instance
                            if all(att in event.vmap and event.vmap[att] == val for att, val in
                                   log.objects[curr_inst_per_type[obj_type]].vmap.items() if
                                   att in object_to_property[obj_type]):
                                event.omap.add(curr_inst_per_type[obj_type])
                        else:
                            previous_events_without_inst[obj_type].append(event)
                """

                # gather instances to add to events with the case object instance and

                """
                for obj_inst in event.omap:
                    if self.log.case_object:
                        if log.type_mapping[obj_inst] == self.log.case_object:
                            # print("add", log.type_mapping[obj_inst], obj_inst)
                            case_object_instances.add(obj_inst)
                        case_instances.add(obj_inst)

            """
            # Assigns all instances found in the case to the events containing the case object type (1:N)
            """
            if self.log.case_object:
                for event in case.events:
                    if len(case_object_instances) == 1:
                        event.omap.add(list(case_object_instances)[0])
                    #if self.log.case_object in event.object_type_to_actions:
                    #    event.omap.update(case_instances)
        for event in log.events:
            if not event.is_duplicate:
                others = log.find_duplicates(event)
                for other in others:
                    other.omap.update(other.omap)


    def assignment_n_to_1(self, log: Log, object_to_property):
        for _, case in log.cases.items():
            #case_instances = set()
            case_object_instances = set()
            for event in case.events:
                """
                # gather instances to add to events with the case object instance and
                """
                for obj_inst in event.omap:
                    if self.log.case_object:
                        if log.type_mapping[obj_inst] == self.log.case_object:
                            # print("add", log.type_mapping[obj_inst], obj_inst)
                            case_object_instances.add(obj_inst)
                        #case_instances.add(obj_inst)

            """
            # Assigns all instances found in the case to the events containing the case object type (1:N)
            """
            if self.log.case_object:
                for event in case.events:
                    if len(case_object_instances) == 1:
                        event.omap.add(list(case_object_instances)[0])
                    #if self.log.case_object in event.object_type_to_actions:
                    #    event.omap.update(case_instances)
        for event in log.events:
            if not event.is_duplicate:
                others = log.find_duplicates(event)
                for other in others:
                    event.omap.update(other.omap)


    def assignment_full(self, log: Log, object_to_property):
        # If an instance was discovered from the specific event e, it is already there.
        # An object instance was discovered from an event f that happened before and is part of the same case c
        # AND the lifecycle of an object instance o in c has completed
        # AND no instance of type(o) was discovered in e.
        regularities = self.get_object_level_behavioral_regularities(log=log)

        for _, case in log.cases.items():
            case_instances = set()
            buffer = {obj_type: {obj_type_2: [] for obj_type_2 in log.objects_per_type.keys()} for obj_type in
                      log.object_types}
            observed_end_events = {obj_type: {obj_type_2: [] for obj_type_2 in log.objects_per_type.keys()} for obj_type
                                   in
                                   log.object_types}
            observed_start_events = {obj_type: {obj_type_2: [] for obj_type_2 in log.objects_per_type.keys()} for
                                     obj_type
                                     in
                                     log.object_types}
            propagation_buffer = {}

            previous_events_without_inst = {obj_type: [] for obj_type in log.objects_per_type.keys()}

            curr_inst_per_type = {obj_type: None for obj_type in log.objects_per_type.keys()}
            case_object_instances = set()

            for event in case.events:
                """
                1:1, 1:N
                we first assign the closest active object instance to events that have the type of that object instance
                but no other instance of that type already
                """
                for obj_type in event.object_type_to_actions.keys():

                    if obj_type not in log.obj_types_from_event_atts:
                        continue
                    if event.label in self.instance_discoverer.obj_type_to_start_events[obj_type]:
                        all_inst = [oi for oi in event.omap if log.type_mapping[oi] == obj_type]
                        if len(all_inst) > 0:
                            curr_inst_per_type[obj_type] = all_inst[0]

                            # Here we handle events that occured previously in the case,
                            # contain a object type for which they do not have an instance.
                            new_prev_list = []
                            for ev in previous_events_without_inst[obj_type]:
                                if all(att in ev.vmap and ev.vmap[att] == val for att, val in
                                       log.objects[curr_inst_per_type[obj_type]].vmap.items() if
                                       att in object_to_property[obj_type]):
                                    ev.omap.add(curr_inst_per_type[obj_type])
                                else:
                                    new_prev_list.append(ev)
                            previous_events_without_inst[obj_type] = new_prev_list
                        else:
                            curr_inst_per_type[obj_type] = None
                    if not any(log.type_mapping[oi] == obj_type for oi in event.omap):
                        if curr_inst_per_type[obj_type]:
                            # check the object properties, if the values differ, it cannot be the same instance
                            if all(att in event.vmap and event.vmap[att] == val for att, val in
                                   log.objects[curr_inst_per_type[obj_type]].vmap.items() if
                                   att in object_to_property[obj_type]):
                                event.omap.add(curr_inst_per_type[obj_type])
                        else:
                            previous_events_without_inst[obj_type].append(event)
                """
                N:M relations

                # gather instances to add to events with the case object instance and
                # add buffer events for heuristic assignment
                # add found object instances that belong to other object instances
                """
                to_add = set()
                for obj_inst in event.omap:
                    if self.log.case_object:
                        if log.type_mapping[obj_inst] == self.log.case_object:
                            # print("add", log.type_mapping[obj_inst], obj_inst)
                            case_object_instances.add(obj_inst)
                        case_instances.add(obj_inst)
                    for obj_type in buffer.keys():
                        if log.type_mapping[obj_inst] in buffer[obj_type]:
                            buffer[obj_type][log.type_mapping[obj_inst]].append(obj_inst)
                    if log.type_mapping[obj_inst] in self.instance_discoverer.obj_type_to_end_events and event.label in \
                            self.instance_discoverer.obj_type_to_end_events[log.type_mapping[obj_inst]]:
                        for obj_type in observed_end_events.keys():
                            observed_end_events[obj_type][log.type_mapping[obj_inst]].append(obj_inst)

                    if log.type_mapping[
                        obj_inst] in self.instance_discoverer.obj_type_to_start_events and event.label in \
                            self.instance_discoverer.obj_type_to_start_events[log.type_mapping[obj_inst]]:
                        for obj_type in observed_start_events.keys():
                            observed_start_events[obj_type][log.type_mapping[obj_inst]].append(obj_inst)
                    if obj_inst in propagation_buffer:
                        to_add.update(propagation_buffer[obj_inst])
                event.omap.update(to_add)

                """
                N:M relations

                # Next, we assign object instances to events that happen after its lifecycle completes
                # (an end event that refers to that object instance occurs before/with the current event
                # if the type of an object is not already part of an event itself
                """
                for obj_type, _ in observed_end_events.items():
                    if obj_type == self.log.case_object:
                        continue
                    if obj_type not in event.object_type_to_actions:

                        for current_obj_type in event.object_type_to_actions.keys():
                            # print(obj_type, current_obj_type, observed_start_events[current_obj_type])

                            # if current_obj_type not in log.object_types or obj_type == current_obj_type or (obj_type, current_obj_type, REVERSE_STRICT_ORD) in regularities:
                            #    continue
                            if (obj_type, current_obj_type, STRICT_ORD) in regularities:
                                # if len(observed_start_events[current_obj_type][obj_type]) > 0:
                                if current_obj_type in self.instance_discoverer.obj_type_to_start_events and event.label in \
                                        self.instance_discoverer.obj_type_to_start_events[current_obj_type]:
                                    # observed_start_events[current_obj_type][obj_type].remove(
                                    #     observed_start_events[current_obj_type][obj_type][0])
                                    # print(obj_type, current_obj_type)
                                    # event.omap.update(buffer[obj_type])
                                    object_inst_for_propagating = [i for i in event.omap if
                                                                   log.type_mapping[i] == current_obj_type]
                                    if len(object_inst_for_propagating) == 1:
                                        object_inst_for_propagating = object_inst_for_propagating[0]
                                        if object_inst_for_propagating not in propagation_buffer:
                                            propagation_buffer[object_inst_for_propagating] = set()
                                    else:
                                        object_inst_for_propagating = None

                                    # if obj_type == "product":
                                    #    print(buffer)
                                    #    print(observed_end_events)
                                    removed = set()
                                    while len(observed_end_events[current_obj_type][
                                                  obj_type]):  # > 0 and len(buffer[current_obj_type][obj_type]) > 0:
                                        # print("end", observed_end_events[current_obj_type][obj_type][0], buffer[current_obj_type][obj_type][0])

                                        # event.omap.add(buffer[current_obj_type][obj_type][0])
                                        event.omap.add(observed_end_events[current_obj_type][obj_type][0])
                                        if object_inst_for_propagating:
                                            propagation_buffer[object_inst_for_propagating].add(
                                                observed_end_events[current_obj_type][obj_type][0])
                                        removed.add(observed_end_events[current_obj_type][obj_type][0])
                                        observed_end_events[current_obj_type][obj_type].remove(
                                            observed_end_events[current_obj_type][obj_type][0])

                                    # In case there is a transitive dependency  item -> package -> delivery and
                                    # items have been assigned to a subsequent type,
                                    # remove them also for the dependent type
                                    # for dependent_obj_type in observed_end_events.keys():
                                    #     if (current_obj_type, dependent_obj_type, STRICT_ORD) in regularities:
                                    #         for remo in removed:
                                    #             if remo in observed_end_events[dependent_obj_type][obj_type]:
                                    #                 observed_end_events[dependent_obj_type][obj_type].remove(remo)
                                    #     # if obj_type == "product":
                                    #    print(event.label, buffer[current_obj_type][obj_type][0])
                                    # buffer[current_obj_type][obj_type].remove(buffer[current_obj_type][obj_type][0])

            """
            # Assigns all instances found in the case to the events containing the case object type (1:N)
            """
            if self.log.case_object:
                for event in case.events:
                    if len(case_object_instances) == 1:
                        event.omap.add(list(case_object_instances)[0])
                    if self.log.case_object in event.object_type_to_actions:
                        event.omap.update(case_instances)
        for event in log.events:
            if not event.is_duplicate:
                others = log.find_duplicates(event)
                for other in others:
                    event.omap.update(other.omap)

    def assign_instances_to_events(self, object_to_property):
        card = self.log.check_cardinalities(self.instance_discoverer)
        if card == N_TO_M:
            self.assignment_full(self.log, object_to_property)
        elif card == ONE_TO_N:
            print("Assigning based on properties and closest occurrence")
            self.assignment_general(self.log, object_to_property)
        else:
            self.assignment_n_to_1(self.log, object_to_property)
