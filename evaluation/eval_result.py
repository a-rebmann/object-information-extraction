from evaluation.metrics import classification_report, conf_matrix


class EvaluationResult:

    def __init__(self, log_name, ocel_name, config):
        self.explicit_ratio = 0

        self.obj_type_precision = 0
        self.obj_type_recall = 0
        self.obj_type_f1 = 0
        self.obj_type_count = 0

        self.inst_to_event_f1 = 0
        self.inst_to_event_recall = 0
        self.inst_to_event_precision = 0
        self.inst_to_event_count = 0

        self.obj_prop_f1 = 0
        self.obj_prop_recall = 0
        self.obj_prop_precision = 0
        self.obj_prop_count = 0

        self.obj_inst_f1 = 0
        self.obj_inst_recall = 0
        self.obj_inst_precision = 0
        self.obj_inst_count = 0


        self.time_s0 = 0
        self.time_s1 = 0
        self.time_s2 = 0
        self.time_s3 = 0
        self.time_s4 = 0
        self.time_s5 = 0

        self.log_name = log_name
        self.ocel_name = ocel_name
        self.config = config
        self.attribute_true = []
        self.attribute_pred = []
        self.object_type_to_att_true = []
        self.object_type_to_att_pred = []
        self.combined_att_type_obj_type_true = []
        self.combined_att_type_obj_type_pred = []

class FullEvaluationResult:

    def __init__(self):
        pass

