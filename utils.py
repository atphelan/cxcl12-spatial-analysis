'''
I/O utils
'''
# %%
import pandas as pd
import os
from typing import List

def identify(path):
    file_lookup = {
        "csv": csvFile,
        "pkl": pklFile,
        "txt": txtFile,
        "json": jsonFile,
    }
    if os.path.isdir(path):
        return Folder(path)
    else:
        ext = path.split('.')[-1]
        return file_lookup.get(ext, File)(path) # Return a file object of the correct type or base type failing that

class File():
    def __init__(self, path: str):
        self.type = path.split('.')[-1]
        self.name = path.split('/')[-1]
        self.path = path

    def __eq__(self, other):
        return self.path == other.path

    def __repr__(self):
        return self.name
    
    def read(self):
        raise NotImplementedError("No reading procedure for this filetype yet")
    
    def write(self, data):
        raise NotImplementedError("No writing procedure for this filetype yet")

    def append(self, data):
        raise NotImplementedError("No append procedure for this filetype yet")    

class csvFile(File):

    def read(self):
        return pd.read_csv(self.path)
    
    def write(self, df: pd.DataFrame, index: bool):
        df.to_csv(self.path, index=index)

    def append(self, df: pd.DataFrame, index: bool):
        self.write(pd.concat([self.read(), df], ignore_index=True), index=index)

class jsonFile(File):

    def read(self):
        return pd.read_json(self.path)
    
    def write(self, df: pd.DataFrame):
        df.to_json(self.path)

    def append(self, df: pd.DataFrame):
        self.write(pd.concat([self.read(), df], ignore_index=True))

class pklFile(File):

    def read(self):
        return pd.read_pickle(self.path)

    def write(self, df):
        df.to_pickle(self.path)

    def append(self, df):
        self.write(pd.concat([self.read(), df], ignore_index=True))

class txtFile(File):

    def read(self):
        with open(self.path) as f:
            contents = []
            for line in f:
                contents.append(line)
        return contents
    
    def write(self, contents: List[str]):

        for i in range(1, len(contents)):
            contents[i] = '\n' + contents[i]
        with open(self.path, 'w') as f:
            f.writelines(contents)
    
    def append(self, contents: List[str]):
        for i in range(0, len(contents)):
            contents[i] = '\n' + contents[i]
        with open(self.path, 'a') as f:
            f.writelines(contents)        
        

class Folder():
    def __init__(self, path: str):
        self.name = path.split('/')[-1]
        self.path = path
        self.items = [identify(os.path.join(path, f)) for f in os.listdir(path)]
        self.subfolders = [f for f in self.items if type(f) == Folder]
        self.files = [f for f in self.items if not f in self.subfolders]

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        return self.path == other.path
    
    def make_folder(self):
        os.mkdir(self.path)

    def get_file_dict(self):
        return {f.name: f for f in self.files}

# %%
if __name__ == '__main__':
    print(os.getcwd())
    print(os.listdir())

    f1 = identify('utils/testfile1.txt')

    print('f1 text: \n', f1.read())
    f1.write(['Newline 1', 'Newline 2'])
    print('f1 text after write: \n', f1.read())
    f1.append(['Appended line 1', 'Appended line 2'])
    print('f1 text after append: \n', f1.read())

    f2 = identify('utils/testfile2.csv')

    df2 = f2.read()
    print(df2)
    f2.write(pd.DataFrame({'a': [1, 2, 3], 'b': [2, 3, 4]}, index=[0,1,2]), index=False)
    print(f2.read())
    f2.append(pd.DataFrame({'a': [4, 4], 'b': [5,4]}, index=[3,4]), index=False)
    print(f2.read())

