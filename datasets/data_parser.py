import argparse
import csv

class DataParser:
    def parse(data_path):
        annotations = {}
        with open(f"{data_path}/Anno/identity_CelebA.txt", "r") as file:
            for line in file.readlines():
                annotations[line.strip().split()[0]] = {"id": line.strip().split()[1]}

        with open(f"{data_path}/Eval/list_eval_partition.txt", "r") as file:
            for line in file.readlines():
                if line.strip().split()[0] in annotations:
                    annotations[line.strip().split()[0]]["partition"] = line.strip().split()[1]
                    
        with open(f"{data_path}/Anno/list_attr_celeba.txt", "r") as f:
            attributes = f.readlines()
            attr_names = attributes[1].strip().split()
            for line in attributes[2:]:
                parts = line.strip().split()
                annotations[parts[0]]["attributes"] = [1 if int(attr) == 1 else 0 for attr in parts[1:]]


        with open(f"{data_path}/annotations.csv", "w", newline="") as csvfile:
            fieldnames = ["filename", "id", "partition"] + attr_names
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for filename, data in annotations.items():
                row = {"filename": filename, "id": data["id"], "partition": data["partition"]}
                row.update({attr_name: attr for attr_name, attr in zip(attr_names, data["attributes"])})
                writer.writerow(row)
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_path", help="Dataset path")
    args = parser.parse_args()

    parser = DataParser()
    parser.parse(args.src_path)