from nltk import WordNetLemmatizer
from const import XES_NAME, XES_TIME, XES_CASE, ONE_TO_N


class PropertyAssigner:

    def __init__(self, config, log, instance_disc):
        self.config = config
        self.log = log
        self.object_to_property = {}
        self.lemmatizer = WordNetLemmatizer()
        self.instance_disc = instance_disc

    def assign_properties(self, log, candidates, instance_attributes):
        taken = set()
        for obj_type, insts in log.objects_per_type.items():

            if obj_type not in candidates or len(insts) == 0:
                continue
            self.object_to_property[obj_type] = set()
            for att in log.attribute_names:
                if att in log.att_names_no_case_atts or att in instance_attributes.values():
                    continue
                if obj_type not in log.obj_types_from_event_atts:
                    if obj_type in log.obj_att_name_matches and att in log.obj_att_name_matches[
                        obj_type] or obj_type in att.lower():
                        print("Name match", att, obj_type)
                        self.object_to_property[obj_type].add(att)
                        taken.add(att)
                    # print(self.extractor.get_unique_vals(att), att)
                    elif len(insts) >= len(
                            self.log.aug_log.df[att].unique()) and obj_type not in log.obj_types_from_event_atts:
                        inst_to_att = {oinst.oid: set([e.vmap[att] for e in oinst.discovered_from if att in e.vmap]) for
                                       oinst in insts}
                        if all(len(val) == 1 for val in inst_to_att.values()):
                            print("Match", att, obj_type)
                            self.object_to_property[obj_type].add(att)
                            #taken.add(att)
            for att in candidates[obj_type]:
                if att in instance_attributes.values() or att in taken or att in [XES_TIME, XES_CASE, XES_NAME]:
                    continue

                print(att, "is property?")
                # inst_to_att = {oinst.oid: set([e.vmap[att] for e in log.events if oinst.oid in e.omap]) for oinst in insts}
                inst_to_att = {oinst.oid: set([e.vmap[att] for e in oinst.discovered_from if att in e.vmap]) for oinst
                               in insts}
                if all(len(val) == 1 for val in inst_to_att.values()):
                    print(att, obj_type, "property!")
                    self.object_to_property[obj_type].add(att)
                    #taken.add(att)
                    for inst in insts:
                        inst.vmap[att] = inst_to_att[inst.oid].pop()
                if self.instance_disc.cardinality == ONE_TO_N and all(len(val) == 0 for val in inst_to_att.values()):
                    print(att, obj_type, "property!")
                    self.object_to_property[obj_type].add(att)
                    #taken.add(att)
        for att in log.attribute_names:
            if att not in taken and "case:" in att and log.case_object and att not in instance_attributes.values():
                if log.case_object not in self.object_to_property:
                    self.object_to_property[log.case_object] = set()
                self.object_to_property[log.case_object].add(att)
        log.attribute_names = set(v for val in self.object_to_property.values() for v in val)
