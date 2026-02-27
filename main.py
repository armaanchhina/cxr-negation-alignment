import json


def format_text(finding, report):
    return f"FINDING {finding} [SEP] REPORT: {report}"

def load_file():
    with open("cxr-align.json", "r") as f:
        raw_data = json.load(f)
    return raw_data


def label_data(data: dict):
    samples = []
    cases = data['mimic']
    print(len(cases.keys()))
    for case_id, case in cases.items():
        chosen = case["chosen"]

        samples.append({
            "text": format_text(chosen, case["report"]),
            "label": 0 
        })

        samples.append({
            "text": format_text(chosen, case["negation"]),
            "label": 1
        })

        samples.append({
            "text": format_text(chosen, case["omitted"]),
            "label": 2
        })

    return samples



def main():
    raw_data = load_file()
    samples = label_data(raw_data)
    print("Total Samples: ", len(samples))

main()