import sys, os
import spacy
import pandas as pd
import numpy as np
import tldextract
import ipaddress
import re
import csv 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta


NOFETCHHANDLE = tldextract.TLDExtract(suffix_list_urls=())
company_db = []
ccadb_list = set([])
ner_map = {}
company_map = {}
nlp = None

SPACY_NER_PATH = "/path/to/en_core_web_trf-3.7.3/"
CCADB_PATH = "/path/to/AllCertificateRecordsReport.csv"
COMPANY_DATA1_PATH = "/path/to/companies-2023-q4-sm.csv"
COMPANY_DATA2_PATH = "/path/to/companies_sorted.csv"


def get_ccadb_list():
    with open(CCADB_PATH, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ccadb_list.add(row[2])

def get_company_names():
    global company_db
    df = pd.read_csv(COMPANY_DATA1_PATH)
    names1 = df['name'].dropna().to_list()
    
    df = pd.read_csv(COMPANY_DATA2_PATH)
    names2 = df['name'].dropna().to_list()
 
    names = set(names1).union(set(names2))

    cleaned_names = set([])
    for name in names:
        n = prep(name)
        cleaned_names.add(n)
    
    company_db = list(cleaned_names)

def standardize_string(s):
    return re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', ' ', s)).strip()

def remove_legal_suffixes(company_name):
    legal_suffixes = [' ltd', ' limited', ' corporation', ' corp', 'llc', 'gmbh', 'inc', 'incorporated']
    name_split = company_name.split(" ")
    
    if len(name_split) < 2:
        return company_name

    if name_split[-1].lower() in legal_suffixes:
        return " ".join(name_split[:-1])
    
    return company_name

def prep(s):
    if not s:
        return ""
    
    name = s.lower()
    if len(name) == 1:
        return name
    
    name = standardize_string(name)
    name = remove_legal_suffixes(name)
    return name

def check_campus_issuer(x):
    if x == ".. campus issuer ..": # we do not provide the exact name of the campus issuer
        return True
    return False


def is_email(teststr):
    email_regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    return bool(re.search(email_regex, teststr))

def is_CA(teststr):
    ca_regex = r'(^|\s)(CA|Certification Authority|Certificate Authority|Encrypt Authority|Trust Authority|Validation Authority)(\s|$)'

    if re.search(ca_regex, teststr, re.IGNORECASE) or teststr in ccadb_list:
        return True

    return False

def is_domain(teststr):
    regdom = NOFETCHHANDLE(teststr).registered_domain
    return bool(regdom)

def is_localhost(teststr):
    return "localhost" in teststr or "localdomain" in teststr

def is_ip(input_str):
    input_str = input_str.replace("-", ".").replace("ip-", "").replace("ip", "")
    try:
        ipaddress.ip_address(input_str)
        return True
    except ValueError:
        return False

def is_mac(input_str):
    return bool(re.match(r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$", input_str))

def clsfy_cert_cn(teststr):
    if not teststr:
        return "Empty"

    if is_email(teststr):
        return "Email"
    elif is_domain(teststr):
        return "Domain"
    elif is_localhost(teststr.lower()):
        return "Localhost"
    elif is_ip(teststr):
        return "IP"
    elif is_mac(teststr.lower()):
        return "MAC"
    elif is_CA(teststr):
        return "CA"

    return "unknown"

def clsfy_cert_san(san_list):
    return_list = []
    for san_ in san_list:
        san = san_[0]
        result = clsfy_cert_cn(san)
        return_list.append([san, result])
    return return_list

def apply_san_result(san_list):
    classes = set([])
    for san_ in san_list:
        classes.add(san_[1])
    return list(classes)

def get_san_ner_company_targets(san_lists):
    target_list = []
    for san_list in san_lists:
        for san in san_list:
            if san[1] == "unknown": target_list.append(prep(san[0]))
    return target_list


def clsfy_company_san(san_list, thres):
    return_list = []
    for san_ in san_list:
        if san_[1] != "unknown":
            return_list.append(san_)
            continue

        san = prep(san_[0])
        if clsfy_company(san, thres):
            result = [san_[0], "Company"]
        else:
            result = [san_[0], "unknown"]
        return_list.append(result)
    return return_list


def handle_ner_result(classes):
    if "PERSON" in classes:
        return "PERSON"
    if "PRODUCT" in classes:
        return "PRODUCT"
    if "ORG" in classes:
        return "ORG"
    return "unknown"


def clsfy_ner_san(san_list):
    return_list = []
    for san_ in san_list:
        if san_[1] != "unknown":
            return_list.append(san_)
            continue

        san = prep(san_[0])
        result_str = clsfy_ner(san)
        if result_str == "None":
            result = "unknown"
        else:
            result = handle_ner_result(result_str.split(","))
        return_list.append([san_[0], result])
    return return_list


def get_matrix(input_words):
    vectorizer = TfidfVectorizer()
    all_texts = company_db + input_words
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    return tfidf_matrix, all_texts

def clsfy_company(teststr, thres):
    thres_map = {0.85: 1, 0.90: 2, 0.95: 3, 0.99: 4}
    return company_map[teststr][thres_map[thres]] == "True"

def read_company_seed():
    global company_map
    paths = ["/path/to/company_sim_seed.csv"]

    for path in paths:
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in company_map:
                    company_map[row[0]] = row[1:]

def prepare_company_clsfy(total_names):
    read_company_seed()
    targets = list(set(total_names) - set(company_map.keys()))

    if not targets:
        return

    tfidf_matrix, all_texts = get_matrix(targets)
    with open("/path/to/company_sim_seed.csv", "a") as f: # save the company classification results as a file for future use
        writer = csv.writer(f)
        for name in targets:
            result = is_company(name, tfidf_matrix, all_texts)
            company_map[name] = [result[1] > 0.85, result[1] > 0.9, result[1] > 0.95, result[1] > 0.99]
            writer.writerow([name] + company_map[name])
            f.flush()

def read_ner_seed():
    paths = ["/path/to/ner_seed.csv"]
    ner_map_tmp = {}

    for path in paths:
        if not os.path.isfile(path):
            continue

        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] not in ner_map_tmp:
                    ner_map_tmp[row[0]] = row[1:]
    return ner_map_tmp


def prepare_ner(total_names):
    ner_map_tmp = read_ner_seed()

    targets = set(total_names) - set(ner_map_tmp.keys())
    print(len(targets))
    
    if len(targets) == 0:
        return ner_map_tmp
    
    path = "/path/to/ner_seed.csv" # save the NER classification results as a file for future use
    with open(path, "a") as f:
        writer = csv.writer(f)
        for name in targets:
            ner_map_tmp[name] = prepare_ner_aux(name)
            writer.writerow([name] + ner_map_tmp[name])
            f.flush()
    return ner_map_tmp   

def clsfy_ner(teststr):
    ret = ner_map[teststr]
    if len(ret) == 0:
        return "None"
    return ",".join(ret)

def apply_ner_result(prev, cls):
    if prev != "unknown":
        return prev

    if cls == "None": return "unknown"
    classes = cls.split(",")
    if "PERSON" in classes:
        return "PERSON"
    if "PRODUCT" in classes:
        return "PRODUCT"
    if "ORG" in classes:
        return "ORG"
    return "unknown"

def apply_company_result(x, is_company):
    if x != "unknown":
        return x
    if is_company:
        return "Company"
    return "unknown"


# post processing
def post_processing(x):
    if x["certificate_issuer_CN"] == "Bloomberg terminal SubCA":
        return True

    if (x["class"] == "PERSON") or (x["class"] == "ORG") or (x["class"] == "PRODUCT"):
        if (x["certificate_issuer_CN"] == x["certificate_subject_CN"]) and (x["certificate_key_length"] == 256.0):
            return True
        if x["certificate_issuer_CN"] == "High Assurance SUDI CA" or x["certificate_issuer_CN"] == "ACT2 SUDI CA":
            return True
        if x["certificate_issuer_O"] == "VIZIO Inc.":
            return True
    if (x["class"] == "PERSON"):
        if len(x["cleaned_cn"]) < 5:
            return True
        if not (" " in x["cleaned_cn"]):
            return True

    if x["cleaned_cn"].isdigit():
        return True
    return False

def find_macbook_pattern(text):
    pattern = r'\w*macbook[\w-]*\.local'
    if re.search(pattern, text.lower()):
        return True
    return False

def find_air_watch_pattern(text):
    if text.endswith("vpn.air-watch.com"):
        return True
    return False


def find_hybrid_runbook_worker_pattern(text):
    if text == "Hybrid Runbook Worker":
        return True
    return False


def find_remaining_product(text):
    if text == None:
        return False
    if find_macbook_pattern(text):
        return True
    if find_air_watch_pattern(text):
        return True
    if text in ["Azure ATP Sensor", "twilio", "Android Keystore Key", "BirdsongClient", "NVIDIA GameStream", "AWS IoT Certificate", "AnyDesk Client", "LiveSwitchm", "rtpengine", "Janus", "CarboniteClient", "syncthing", "CIRA Certificate", "FbWebRtc"]:
        return True
    if text.startswith("NIS WindowsAgent"):
        return True
    if text.startswith("v1.LenovoAutoclaimed"):
        return True
    if text.startswith("NX Prod 1"):
        return True
    if find_hybrid_runbook_worker_pattern(text):
        return True
    return False

def find_LANDesk_pattern(text):
    if text == None:
        return False
    if text.startswith("LANDesk"):
        return True
    return False

def find_jamf_pattern(text):
    if text == None:
        return False
    if text.startswith("JamfProtect Client"):
        return True
    return False


def find_tanium_pattern(text):
    if text == None:
        return False
    if text.startswith("taniumzoneserver"):
        return True
    return False

def find_ocitanium_pattern(text):
    if text == None:
        return False
    if text.startswith("ocitaniumzoneserver"):
        return True
    return False

def find_thirdwayv_pattern(text):
    if text == None:
        return False
    if text == "Thirdwayv Inc":
        return True
    return False


def find_apple_id_pattern(text):
    if text == None:
        return False
    if text.startswith("com.apple.idms.appleid.prd."):
        return True
    return False


def find_remain_ip_pattern(text):
    if text == None:
        return False
    if text.startswith("ip-"):
        return True
    return False

def find_nintento_pattern(text):
    if text == None:
        return False
    if text.startswith("NX Prod"):
        return True
    return False

def filter_non_name(class_, name, issuer):

    if class_ != "PERSON":
        return class_

    if "certificate" in name.lower():
        return "unknown"
    if "-" in name or "\\" in name:
        return "unknown"
    return class_


def filter_non_product(class_, name, issuer):
    if class_ != "PRODUCT":
        return class_
    if len(name) == 8 or len(name) == 9:
        return "unknown"
    return class_

def filter_non_company(class_, name, issuer):
    if class_ != "Company":
        return class_
    if name == "__transfer__":
        return "unknown"
    if name == "EMAIL":
        return "unknown"
    return class_


def find_webrtc_pattern(text):
    if text == "WebRTC" or text == "webrtc":
        return True
    return False


def check_san_campus_person(san, name):
    name_split = name.lower().split()
    for word in name_split:
        if (word in san) and (len(word) > 2):
            return True
    return False


def san_postprocess_campus_person(san, cn):
    if check_san_campus_person(san, cn):
        return "Campus_person"
    if find_macbook_pattern(san):
        return "Campus_product"
    return None


def san_postprocess(x):
    return_list = []
    san_dns_list = x["san_dns_list"]
    for san_ in san_dns_list:
        san = san_[0]
        if find_air_watch_pattern(san):
            return_list.append([san, "PRODUCT"])
        elif san.startswith("syncthing"):
            return_list.append([san, "PRODUCT"])
        elif find_ocitanium_pattern(san) or find_tanium_pattern(san):
            return_list.append([san, "Company"])
        elif san.endswith("traefik.default"):
            return_list.append([san, "Company"])
        elif san.startswith("sonos-") and san_[1] != "Domain":
            return_list.append([san, "Company"])
        elif san.startswith("sip:"):
            return_list.append([san, "sip"])
        elif san.startswith("guardicore"):
            return_list.append([san, "PRODUCT"])
        elif (x["class"] == "Company") and (x["certificate_issuer_CN"] != None) and (x["certificate_issuer_CN"].startswith("LANDesk")):
            if san == x["certificate_subject_CN"]:
                return_list.append([san, x["class"]])
            elif san in x["certificate_subject_CN"]:
                return_list.append([san, "unknown"])
            else:
                return_list.append(san_)
        elif x["class"] == "Campus_person":
            result = san_postprocess_campus_person(san, x["certificate_subject_CN"])
            if result == None:
                if san_[1] == "PERSON":
                    return_list.append([san_[0], "unknown"])
                elif san_[1] == "ORG":
                    return_list.append([san_[0], "unknown"])
                elif san_[1] == "PRODUCT":
                    return_list.append([san_[0], "Campus_product"])
                else:
                    return_list.append(san_)
            else:
                return_list.append([san, result])
        else:
            if find_macbook_pattern(san):
                return_list.append([san, "PRODUCT"])
            elif san_[1] == "PERSON" and (len(prep(san)) < 5 or prep(san).isdigit()):
                return_list.append([san, "unknown"])
            else:
                return_list.append(san_)
    return return_list



def count_class(x):
    clss = ["Domain", "Email", "IP", "MAC", "sip", "Localhost", "PERSON", "Campus_person", "Gov_person",  "Campus_ID", "CA", "ORG", "Company", "PRODUCT", "unknown", "Empty"]
    for cls_ in clss:
        n = x.apply(lambda x: cls_ == x).sum()
        print(cls_+",", n)
    print("Total:", len(x))

def count_class_san(x):
    clss = ["Domain", "Email", "IP", "MAC", "sip", "Localhost", "PERSON", "Campus_person", "Gov_person", "Campus_ID", "CA", "ORG", "Company", "CA_ORG_Company", "PRODUCT", "Campus_product", "PRODUCT_ORG_total", "unknown"]
    for cls_ in clss:
        if cls_ == "CA_ORG_Company":
            n = x.apply(lambda x: len( set(["CA", "ORG", "Company"]).intersection(set(x)) ) > 0).sum()
        elif cls_ == "PRODUCT_ORG_total":
            n = x.apply(lambda x: len( set(["CA", "ORG", "Company", "PRODUCT", "Campus_product"]).intersection(set(x)) ) > 0).sum()
        else:
            n = x.apply(lambda x: cls_ in x).sum()
        print(cls_+",", n)
    n = x.apply(lambda x: x == []).sum()
    print("Empty,", n)

    print("Total,", x.apply(lambda x: x != []).sum())


def load_certificate_data():
    path = "/path/to/certificate_dataset" # we do not provide the raw dataset due to privacy issues
    # df = load data from path
    return df

# Main function to process data
def process_data():
    global ner_map

    targets_df = load_certificate_data()

    targets_df["cleaned_cn"] = targets_df["certificate_subject_CN"].apply(lambda x: prep(x))
    targets_df['class'] = targets_df['certificate_subject_CN'].apply(lambda x: clsfy_cert_cn(x))

    # Handle special cases
    targets_df["class"] = targets_df.apply(lambda x: "Campus_person" if check_campus_issuer(x["certificate_issuer_CN"]) else x["class"], axis=1)
    targets_df["class"] = targets_df.apply(lambda x: "Campus_ID" if check_campus_issuer(x["certificate_issuer_CN"]) else x["class"], axis=1)

    # Apply NER
    ner_targets = targets_df[targets_df["class"] == "unknown"]["cleaned_cn"].to_list()
    ner_map = prepare_ner(ner_targets)
    targets_df['class_ner'] = targets_df.apply(lambda x: clsfy_ner(x["cleaned_cn"]) if x["class"] == "unknown" else "None", axis=1)
    targets_df["class"] = targets_df.apply(lambda x: apply_ner_result(x["class"], x["class_ner"]), axis=1)

    # Apply company classification
    company_targets = targets_df[targets_df["class"] == "unknown"]["cleaned_cn"].to_list()
    prepare_company_clsfy(company_targets)
    thres = 0.90
    targets_df['class_company'] = targets_df.apply(lambda x: clsfy_company(x["cleaned_cn"], thres) if x["class"] == "unknown" else False, axis=1)
    targets_df["class"] = targets_df.apply(lambda x: apply_company_result(x["class"], x["class_company"]), axis=1)
    print("\n- [CN] Apply NER & Company classification\n", targets_df['class'].value_counts())

    # Post-processing
    targets_df["class"] = targets_df.apply(lambda x: "unknown" if post_processing(x) else x["class"], axis=1)
    targets_df["class"] = targets_df.apply(lambda x: "IP" if (find_remain_ip_pattern(x["certificate_subject_CN"])) else x["class"], axis=1)
    targets_df["class"] = targets_df.apply(lambda x: filter_non_name(x["class"], x["certificate_subject_CN"], x["certificate_issuer_CN"]), axis=1)
    targets_df["class"] = targets_df.apply(lambda x: filter_non_company(x["class"], x["certificate_subject_CN"], x["certificate_issuer_CN"]), axis=1)
    targets_df["class"] = targets_df.apply(lambda x: "PRODUCT" if find_remaining_product(x["certificate_subject_CN"]) else x["class"], axis=1)
    targets_df["class"] = targets_df.apply(lambda x: "PRODUCT" if find_webrtc_pattern(x["certificate_subject_CN"]) else x["class"], axis=1)

    targets_df["class"] = targets_df.apply(lambda x: "unknown" if ((x["class"] in ["PRODUCT"]) and (x["certificate_subject_CN"] in ["Janus"])) else x["class"], axis=1)
    targets_df["class"] = targets_df.apply(lambda x: "sip" if (x["certificate_subject_CN"] != None) and (x["certificate_subject_CN"].startswith("sip:")) else x["class"], axis=1)
    targets_df["class"] = targets_df.apply(lambda x: "PRODUCT" if (x["certificate_subject_CN"] != None) and (x["certificate_subject_CN"] in ["libdatachannel", "Yeti", "Kurento"]) else x["class"], axis=1)
    targets_df["class"] = targets_df.apply(lambda x: "PRODUCT" if (x["certificate_subject_CN"] != None) and (x["certificate_subject_CN"].startswith("mediasoup")) else x["class"], axis=1)

    print("\n- [CN] Post-processing\n", targets_df['class'].value_counts())


    # SAN DNS
    targets_df["san_dns_list"] = targets_df['san_dns_str'].apply(lambda x: [[san] for san in x.split(",")] if (x != "") & (x != None) else [])
    targets_df["san_dns_list"] = targets_df['san_dns_list'].apply(lambda x: clsfy_cert_san(x))
    tmp = targets_df["san_dns_list"].apply(lambda x: apply_san_result(x))

    san_dns_ner_targets = get_san_ner_company_targets(targets_df["san_dns_list"].to_list())
    ner_map = prepare_ner(san_dns_ner_targets)
    targets_df["san_dns_list"] = targets_df['san_dns_list'].apply(lambda x: clsfy_ner_san(x))
    tmp = targets_df["san_dns_list"].apply(lambda x: apply_san_result(x))

    san_dns_company_targets = get_san_ner_company_targets(targets_df["san_dns_list"].to_list())
    prepare_company_clsfy(san_dns_company_targets)
    thres = 0.90
    targets_df["san_dns_list"] = targets_df['san_dns_list'].apply(lambda x: clsfy_company_san(x, thres))
    tmp = targets_df["san_dns_list"].apply(lambda x: apply_san_result(x))
    print("\n [SAN DNS] Apply NER & Company classification\n", tmp.value_counts())

    targets_df["san_dns_list"] = targets_df.apply(lambda x: san_postprocess(x), axis=1)
    targets_df["class_san_dns"] = targets_df["san_dns_list"].apply(lambda x: apply_san_result(x))
    print("\n [SAN DNS] Post-processing\n", tmp.value_counts())

    print("\n[ Print Stats.. ]\n")
    result_df = targets_df[targets_df["isIssuerOConductInterception"] == False]
    result_publicCA = .. # filter certificates issued by public CA
    result_privateCA = .. # filter certificates issued by private CA


    # Print stats
    print("- [CN] Public CA\n")
    count_class(result_publicCA["class"])

    print("- [CN] Private CA\n")
    count_class(result_privateCA["class"])

    print("- [SAN] Public CA\n")
    count_class_san(result_publicCA["class_san_dns"])
    print("- [SAN] Private CA\n")
    count_class_san(result_privateCA["class_san_dns"])


if __name__ == "__main__":
   
    get_company_names() # load company name dataset
    get_ccadb_list() # load CCADB list
    nlp = spacy.load(SPACY_NER_PATH) # load NER model
    process_data()


