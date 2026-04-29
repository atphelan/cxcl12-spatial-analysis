import pandas as pd
from utils import Folder

def remove_duplicates(target_directory, tol):
    for file in target_directory.files:
        if (file.type == 'csv' and 'analysed' not in file.path) and ('Slide' in file.path and 'duplicates_removed' not in file.path):
            df = file.read()
            df['processed XY'] = [f"{round(row['X']/tol, 0)} {round(row['Y']/tol, 0)}" for i, row in df.iterrows()]
            df.drop_duplicates(subset='processed XY', inplace=True)
            df.drop(columns=['Unnamed: 0', 'processed XY'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            df.sort_values(['X', 'Y'], inplace=True)
            df.to_csv(file.path[:-4] + '_duplicates_removed.csv')

def report_duplicates(target_directory, tol):
    for file in target_directory.files:
        if (file.type == 'csv' and 'analysed' not in file.path) and ('Slide' in file.path and 'duplicates_removed' not in file.path):
            df = file.read()
            df['processed XY'] = [f"{round(row['X']/tol, 0)} {round(row['Y']/tol, 0)}" for i, row in df.iterrows()]
            df.sort_values(['X', 'Y'], inplace=True)
            print(df.groupby(['X', 'Y'], as_index=False).size()['size'].min(), df.groupby(['X', 'Y'],as_index=False).size()['size'].max())
            print(file.path, len(df), len(df['processed XY'].unique()))


if __name__ == '__main__':
    target_directory = Folder('data')
    tol = 1
    # report_duplicates(target_directory, tol)
    remove_duplicates(target_directory, tol)