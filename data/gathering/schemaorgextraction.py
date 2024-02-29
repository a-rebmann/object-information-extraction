import os


def read_and_extract(direct):
    import pandas as pd
    dft = pd.read_csv(os.path.join(direct, 'schemaorgtypes.csv'))
    dfp = pd.read_csv(os.path.join(direct, 'schemaorgproperties.csv'))
    # ACTORS
    actor_terms = ["Organization"]
    actor_terms.extend(
        [a for a in list(dft[(dft["subTypeOf"] == "https://schema.org/Organization")]["label"]) if a not in ["Task", "Activity", "Action"]])
    actor_terms.remove("Project")
    actor_terms.extend(
        [a for a in
         list(dfp[dfp["rangeIncludes"].fillna("missing").str.contains("https://schema.org/Organization")]["label"]) if a not in ["Task", "Activity", "Action"]])


    # OBJECTS

    obj_terms = ["Product", "Products", "Project"]
    obj_terms.extend(
        [a for a in list(dft[(dft["subTypeOf"] == "https://schema.org/Product")]["label"])if a not in ["Task", "Activity", "Action"]])
    obj_terms.extend(
        [a for a in list(dft[(dft["subTypeOf"] == "https://schema.org/Intangible")]["label"])if a not in ["Task", "Activity", "Action"]])


    # ACTION
    act_terms = ["Action", "Task", "Activity"]
    act_terms.extend(
        [a.replace("Action", "") for a in list(dft[(dft["subTypeOf"] == "https://schema.org/Action")]["label"])])
    #print(act_terms)

    # ACTION STATUS
    action_status_terms = []
    action_status_terms.extend([a.replace("Status", "").replace("Action", "") for a in
                                list(dft[(dft["subTypeOf"] == "https://schema.org/ActionStatusType")]["label"])])
    # print(action_status_terms)

    # OBJECT STATUS
    obj_status_terms = ["Status"]

    potential_terms = [a.replace("Order", "") for a in
                       list(dft[(dft["subTypeOf"] == "https://schema.org/OrderStatus")]["label"])]
    obj_status_terms.extend([prop for prop in potential_terms if not any(p in obj_terms for p in prop.split(" "))])

    potential_terms = [a.replace("Payment", "") for a in
                       list(dft[(dft["subTypeOf"] == "https://schema.org/PaymentStatusType")]["label"])]
    obj_status_terms.extend([prop for prop in potential_terms if not any(p in obj_terms for p in prop.split(" "))])

    potential_terms = [a.replace("Reservation", "") for a in
                       list(dft[(dft["subTypeOf"] == "https://schema.org/ReservationStatusType")]["label"])]
    obj_status_terms.extend([prop for prop in potential_terms if not any(p in obj_terms for p in prop.split(" "))])
    potential_terms = [a for a in list(dft[(dft["subTypeOf"] == "https://schema.org/GameServerStatus")]["label"])]
    obj_status_terms.extend([prop for prop in potential_terms if not any(p in obj_terms for p in prop.split(" "))])
    # print(obj_status_terms)
    # OBJECT PROPERTIES
    obj_property_terms = []
    potential_terms = [b.split("/")[-1] for b in
                       [a.split(",") for a in
                        list(dft[(dft["id"] == "https://schema.org/Product")]["properties"])][0]]
    obj_property_terms.extend([prop for prop in potential_terms if not any((p in obj_terms or p in actor_terms) for p in prop.split(" "))])

    potential_terms = [b.split("/")[-1] for b in
                       [a.split(",") for a in
                        list(dft[(dft["id"] == "https://schema.org/Intangible")]["properties"])][0]]
    obj_property_terms.extend([prop for prop in potential_terms if not any((p in obj_terms or p in actor_terms) for p in prop.split(" "))])

    #print(obj_property_terms)

    # dfp = pd.read_csv("../" + DEFAULT_RES_DIR + 'schemaorgproperties.csv')
    # dfp = dfp[(dfp["subTypeOf"] == "https://schema.org/Organization")]
    # print(dfp["label"])
    return actor_terms, act_terms, action_status_terms, obj_terms, obj_status_terms, obj_property_terms


if __name__ == '__main__':
    from main import DEFAULT_RES_DIR

    read_and_extract(DEFAULT_RES_DIR)
