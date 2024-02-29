class Config:

    def __init__(self, input_dir, model_dir, resource_dir, output_dir, bo_ratio, conf_thresh, instance_thresh,
                 exclude_data_origin, we_file, matching_mode, mode, res_type,
                 mask_attribute=None, mask_attribute_name=None):
        if mask_attribute_name is None:
            mask_attribute_name = []
        if mask_attribute is None:
            mask_attribute = []
        self.input_dir = input_dir
        self.model_dir = model_dir
        self.resource_dir = resource_dir
        self.output_dir = output_dir
        self.bo_ratio = bo_ratio
        self.conf_thresh = conf_thresh
        self.instance_thresh = instance_thresh
        self.word_embeddings_file = we_file
        self.matching_mode = matching_mode
        self.exclude_data_origin = exclude_data_origin
        self.mode = mode
        self.resource_type = res_type
        self.mask_attribute = mask_attribute
        self.mask_attribute_name = mask_attribute_name

    def __repr__(self):
        masks = "-".join([mask for mask in self.mask_attribute])
        return f'(masked={masks})'
