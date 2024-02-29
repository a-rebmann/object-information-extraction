import os
import readwrite.loader as load

# WIP OCEL folder for testing
DEFAULT_OCEL_INPUT_DIR = '../../input/evaluation/ocel/'

def load_ocel_logs():
    list_of_files = {}
    file_to_data = {}
    for (dir_path, dir_names, filenames) in os.walk(DEFAULT_OCEL_INPUT_DIR):
        for filename in filenames:
            if filename.endswith('ocel'):
                list_of_files[filename] = os.sep.join([dir_path])
    for key, value in list_of_files.items():
        file_to_data[key] = set(), set(), set()
        oclog = load.load_ocel(value, key)
        print(key)
        for event in oclog["ocel:events"]:
            file_to_data[key][0].add(oclog["ocel:events"][event]["ocel:activity"])
        for obj in oclog["ocel:global-log"]["ocel:object-types"]:
            file_to_data[key][1].add(obj)
        if "ocel:attribute-names" in oclog["ocel:global-log"].keys():
            for obj in oclog["ocel:global-log"]["ocel:attribute-names"]:
                file_to_data[key][2].add(obj)
        print(file_to_data[key])

if __name__ == '__main__':
    load_ocel_logs()