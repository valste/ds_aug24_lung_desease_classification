# -*- coding: utf-8 -*-
import os
import shutil

import pandas as pd


def copy_raw_data(kaggle_path, raw_data_path) -> None:
    """
    Copy raw data from a directory to the data/raw directory.

    Input:
    kaggle_path: str: Path to the directory containing Kaggle data
    raw_data_path: str: Path to the data/raw directory

    Output:
    None
    """
    # Remove existing raw data directory
    if os.path.exists(raw_data_path):
        shutil.rmtree(raw_data_path)

    # Files are in a subdirectory, so copy all files to data/raw
    os.listdir(kaggle_path)
    shutil.copytree(kaggle_path, raw_data_path)


def convert_excel_to_csv(path) -> None:
    """
    Convert Excel files to CSV files.

    Input:
    path: str: Path to the directory containing Excel files

    Output:
    None
    """
    # List all files in the folder
    excel_files = [f for f in os.listdir(path) if f.endswith(".xlsx") or f.endswith(".xls")]

    # Convert each Excel file to CSV
    for excel_file in excel_files:
        excel_path = os.path.join(path, excel_file)
        xls = pd.ExcelFile(excel_path)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            csv_file_name = f"{os.path.splitext(excel_file)[0]}.csv"
            csv_path = os.path.join(path, csv_file_name)
            df.to_csv(csv_path, index=False)
            print(f"Converted {excel_file} - to {csv_file_name}")
            os.remove(excel_path)
            print(f"{excel_file} has been removed")


def create_class_folders(path, categories, classes) -> None:
    """
    Create class folders for each category.

    Input:
    path: str: Path to the directory to create class folders
    categories: list: List of categories
    classes: list: List of classes
    """
    for category in categories:
        for cls in classes:
            target = os.path.join(path, category, cls)
            print(f"Creating directory '{target}'...")
            try:
                os.makedirs(target)
                print(f"Directory '{target}' created successfully.")
            except FileExistsError:
                print(f"Directory '{target}' already exists.")
            except PermissionError:
                print(f"Permission denied: Unable to create '{target}'.")
            except Exception as e:
                print(f"An error occurred: {e}")


def move_files_to_class_folders(old_path, new_path, categories, classes, binary=False) -> None:
    """
    Move files to class folders for
    each category.

    Input:
    old_path: str: Path to the directory containing files
    new_path: str: Path to the directory to move files
    categories: list: List of categories
    classes: list: List of classes
    binary: bool: If True, move files to a single "Sick" folder

    Output:
    None
    """
    for category in categories:
        for cls in classes:
            source = os.path.join(old_path, cls, category)
            target = (
                os.path.join(new_path, category, "Sick")
                if binary
                else os.path.join(new_path, category, cls)
            )
            try:
                print(f"Moving files from '{source}' to '{target}'...")
                shutil.copytree(source, target, dirs_exist_ok=True)
                print("Files copied successfully.")
            except FileExistsError:
                print("Files already copied.")
            except PermissionError:
                print("Permission denied: Unable to copy files.")
            except Exception as e:
                print(f"An error occurred: {e}")

            fimages = os.listdir(target)
            print(f"Number of files: {len(fimages)} from '{source}' to '{target}'...")
