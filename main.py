import json
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from cxr_dataset import CXRTextDataset
from torch.utils.data import DataLoader


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

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=42,
        stratify=[s["label"] for s in samples] # keeps the label proportions same
    )

    print(f"Train size: {len(train_samples)}")
    print(f"Val size: {len(val_samples)}")

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    train_dataset = CXRTextDataset(train_samples, tokenizer)
    val_dataset = CXRTextDataset(val_samples, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

main()