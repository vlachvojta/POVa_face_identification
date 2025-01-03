import os
from collections import defaultdict
from typing import Optional
import csv

from clearml import Task

class TrainingMonitor:
    def __init__(
        self,
        name: Optional[str]=None,
        project_name: Optional[str]=None,
        output_path: Optional[str] = 'monitor_output'
    ):
        self.iterations: list[int] = []
        self.values: dict[str, list[float]] = defaultdict(list)

        self.name = name if name is not None else "training"
        self.project_name = project_name
        self.output_path = output_path

        self.log_name = f"{self.project_name}_{self.name}" if self.project_name is not None else self.name
        self.log_path = os.path.join(self.output_path, f"{self.log_name}.csv")
        if os.path.exists(self.log_path):
            self.load_from_csv(self.log_path)

        if project_name is not None and name is not None:
            self.task = Task.init(project_name=project_name, task_name=name, continue_last_task=0)
        else:
            self.task = None

    def add_value(
        self,
        title: str,
        series: str,
        value: float,
    ) -> None:
        key = f"{title}_{series}"
        self.values[key].append(value)
        if self.task is not None:
            self.task.logger.report_scalar(title=title, series=series, value=value, iteration=self.iterations[-1])

    def report_results(self, digits: int=4) -> None:
        if self.task is None:
            return

        for k, v in self.values.items():
            if "loss" in k:
                self.task.logger.report_single_value(k, round(min(v), digits))
            elif "acc" in k:
                self.task.logger.report_single_value(k, round(max(v), digits))
            else:
                continue

    def save_csv(self, path: str = None) -> None:
        if path is None:
            path = self.log_path

        with open(path, "w") as f_csv:
            keys = sorted(list(self.values.keys()))
            writer = csv.writer(f_csv)
            writer.writerow(["iteration"] + keys)
            for i, it in enumerate(self.iterations):
                writer.writerow([it] + [self.values[k][i] for k in keys])

    def load_from_csv(self, path: str = None) -> None:
        print(f'Loading previous log from {path}')
        if path is None:
            path = self.log_path

        # load csv using csv reader
        with open(path, "r") as f_csv:
            spamreader = csv.reader(f_csv, delimiter=',', quotechar='|')
            keys = next(spamreader)[1:]
            print(f'loaded keys: {keys}')

            for row in spamreader:
                self.iterations.append(int(row[0]))
                for i, k in enumerate(keys):
                    self.values[k].append(float(row[i+1]))

        print(f'Loaded {len(self.iterations)} results, last iteration: {self.iterations[-1]}')

    def get_last_string(self):
        return f"ITER {self.iterations[-1]} " + " ".join([f"{k} {v[-1]:.4f}" for k, v in self.values.items()])