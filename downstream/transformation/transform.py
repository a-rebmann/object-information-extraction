from const import *
from copy import deepcopy
from model.log.log import Log

ocel_events_stub = {
    "ocel:id": {
        "ocel:activity": "",
        "ocel:timestamp": "",
        "ocel:omap": [

        ],
        "ocel:vmap": {

        }
    }
}

ocel_object_stub = {
    "ocel:id": {
        "ocel:type": "",
        "ocel:ovmap": {

        }
    }
}

ocel_stub = {
    "ocel:global-event": {
        "ocel:activity": "__INVALID__"
    },
    "ocel:global-object": {
        "ocel:type": "__INVALID__"
    },
    "ocel:global-log": {
        "ocel:attribute-names": [

        ],
        "ocel:object-types": [

        ],
    },
    "ocel:events": {},
    "ocel:objects": {}
}

def create_ocel_from_log(log: Log):
    ocel_dict = deepcopy(ocel_stub)
    ocel_dict["ocel:global-log"]["ocel:object-types"] = list(log.object_types)
    ocel_dict["ocel:global-log"]["ocel:attribute-names"] = list(log.attribute_names)
    event_id = 1
    for event in log.events:
        if event.is_duplicate:
            continue
        ocel_event = {OCEL_ACTIVITY: event.label,
                      OCEL_TIMESTAMP: event.timestamp,
                      OCEL_OMAP: [str(oinst) for oinst in event.omap],
                      OCEL_VMAP: event.vmap}
        ocel_dict[OCEL_EVENTS][event_id] = ocel_event
        event_id = event_id + 1
    for obj_id, obj in log.objects.items():
        ocel_dict["ocel:objects"][obj_id] = {OCEL_TYPE: obj.obj_type, OCEL_OVMAP: obj.vmap}
    return ocel_dict



