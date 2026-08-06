"""
file_utils.py

Helper functions for reading and writing files.
"""

import json
import os
import pandas as pd


def create_folder(folder_path):

    os.makedirs(folder_path, exist_ok=True)


def save_json(data, save_path):

    with open(save_path, "w") as file:

        json.dump(data, file, indent=4)


def load_json(file_path):

    with open(file_path, "r") as file:

        return json.load(file)


def save_csv(dataframe, save_path):

    dataframe.to_csv(save_path, index=False)


def read_csv(file_path):

    return pd.read_csv(file_path)


def list_images(folder):

    extensions = (".jpg", ".jpeg", ".png")

    return [

        os.path.join(folder, file)

        for file in os.listdir(folder)

        if file.lower().endswith(extensions)

    ]