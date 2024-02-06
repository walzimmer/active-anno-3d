import os
import numpy as np
import json

class OpenLABEL():
    
    def __init__(self, json_path=None, annotator=None, schema_version=None, file_version=None):

        self.annotator = annotator
        self.schema_version = schema_version
        self.file_version = file_version
        self.timestamp = None

        if json_path:
            with open(json_path, 'rb') as f:
                self.json_schema = json.load(f)
        else:
            self.json_schema = self.create_json_schema(annotator,
                                                       schema_version, 
                                                       file_version)

    
    def create_json_schema(self, annotator=None, schema_version=None, file_version=None):
        schema = {
            "openlabel": {
                "metadata": {
                    "annotator": self.annotator if annotator is None else annotator,
                    "file_version": self.file_version if file_version is None else file_version,
                    "schema_version": self.schema_version if schema_version is None else schema_version
                },
                "coordinate_systems": {},
                "frames": {},
                "ontologies": {}
            }
        }
        if self.timestamp is not None:
            self.timestamp = None
            
        return schema


    def save_json(self, save_path):

        assert save_path is not None, "where to save your annotation file?!"
        os.makedirs(save_path, exist_ok=True)

        out_file = os.path.join(save_path, self.timestamp + '.json')
        with open(out_file, 'w') as f:
            json.dump(self.json_schema, f)