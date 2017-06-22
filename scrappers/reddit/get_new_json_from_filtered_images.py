import json
import os

FILTERED_IMAGES_PATH = "images_filtered"
BASE_JSON = "data_cleaned.json"
OUT_FILTERED_JSON = "filtered_data.json"

class JsonGenerator():
    def __init__(self):
        self.ids = []
        self.base_metadata = []
        self.get_ids_from_path()
        self.load_metadata()

    def get_ids_from_path(self):
        for filename in os.listdir(FILTERED_IMAGES_PATH):
            self.ids.append( os.path.splitext(filename)[0] )

    def load_metadata(self):
        with open(os.path.join(BASE_JSON)) as data_file:
            self.base_metadata = json.load(data_file)

    def generate_new_json(self):
        new_data = [x for x in self.base_metadata if x['id'] in self.ids]
        if os.path.exists(OUT_FILTERED_JSON):
            print("{} already exists, nothing done.".format(OUT_FILTERED_JSON))
            return

        with open(OUT_FILTERED_JSON, 'w') as outfile:
            json.dump(new_data, outfile, indent=4, sort_keys=True)


if __name__ == "__main__":
    generator = JsonGenerator()
    generator.generate_new_json()


