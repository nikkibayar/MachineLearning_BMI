import os
import json
OUTPUT_FOLDER="json_out"

def merge():
    if os.path.exists(OUTPUT_FOLDER) is False:
        print("Not Files found, nothing to be merged")
        return

    all_submissions = []
    for filename in os.listdir(OUTPUT_FOLDER):
        if filename.endswith(".json") :
            with open(os.path.join(OUTPUT_FOLDER, filename)) as data_file:
                data = json.load(data_file)
                all_submissions.extend( data )

    outfilename = "data.json"
    if os.path.exists(outfilename):
        print("{} already exists, nothing done.".format(outfilename))
        return

    with open(outfilename, 'w') as outfile:
        json.dump(all_submissions, outfile, indent=4, sort_keys=True)

def convert_kgs_to_lbs(weight_kgs):
    weight_kgs = weight_kgs.lower()
    value = weight_kgs[:weight_kgs.index("kg")]
    return str(int(float(value)*2.2046))

def remove_unit_tag(word):
    word = word.lower()
    value = word[:word.index("lb")]
    return str(value)

def normalize_weight():
    outfilename = "data_cleaned.json"
    if os.path.exists(outfilename) is False:
        print("{} doesn't exist, nothing done.".format(outfilename))
        return
    with open(outfilename) as data_file:
        data = json.load(data_file)

    cleaned_data = []
    for entry in data:
        try:
            float(entry["after_weight"])
            float(entry["before_weight"])

            # if "kg" in entry["after_weight"].lower():
            #     entry["after_weight"] = convert_kgs_to_lbs(entry["after_weight"])
            # if "kg" in entry["before_weight"].lower():
            #     entry["before_weight"] = convert_kgs_to_lbs(entry["before_weight"])
            #
            # if "lb" in entry["after_weight"].lower():
            #     entry["after_weight"] =remove_unit_tag( entry["after_weight"] )
            # if "lb" in entry["before_weight"].lower():
            #     entry["before_weight"] =remove_unit_tag( entry["before_weight"] )



        except Exception as e:
            print("id: {} .Exception: {}".format(entry["id"], e))

        cleaned_data.append(entry)

    outfilename = "data_cleaned.json"
    with open(outfilename, 'w') as outfile:
        json.dump(cleaned_data, outfile, indent=4, sort_keys=True)




# merge()
normalize_weight()