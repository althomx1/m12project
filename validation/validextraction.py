import pandas as pd
from collections import defaultdict

import re
import zipfile
import json

def extract_descriptions():

    map = defaultdict(list)

    path = "C:/Users/Alex Toma/Downloads/nvdcve-1.1-modified.json.zip"


    with zipfile.ZipFile(path,'r') as files:
        with files.open("nvdcve-1.1-modified.json") as js:
            data = json.load(js)

            i = 0
            for cve in data.get("CVE_Items",{}):
                cve_id = cve.get("cve", {}).get("CVE_data_meta", {}).get("ID", None)

                cve_description = cve.get("cve", {}).get("description", {}).get("description_data", [])

                if cve_description:
                    for elem in cve_description:
                        dsc = elem.get("value",None)
                        if dsc:
                            map[cve_id] = dsc
                            i += 1
                if i == 400:
                    break


    df = pd.DataFrame(map.items(),columns=["cve_id", "description"])


    df.to_csv("cve_description_nvd_2.csv", index=False)

extract_descriptions()