'''
Author: Pierce Lane
KUID: 3050467
Date: 08/22/2022
Last modified: 10/27/2022
Purpose: creates a file object that the user can read/write to.
'''
class File():
    #takes in input .txt file name and stores the contents in a string called self._file
    #automatically strips the file if strip is True
    def __init__(self, file_name, /, *, create_file = False, strip = False):
        """
        Creates a file that you can read/write to.
        file_name (required) is a string dictating the name of the file.
        strip (optional) is a bool dictating whether to strip white space off the file on creation. Default False.
        create_file (optional) is a bool dictating whether to create a file named file_name or to use an existing file. Default False.
        """
        self._file_name = file_name
        self._file = ""
        if not create_file:
            with open(self._file_name, "r", encoding='utf-8') as file:
                for line in file:
                    self._file += line #copy whatever's in the actual file to a string
        else:
            with open(self._file_name,"w", encoding='utf-8') as file:
                file.write("")
        
        if strip == True:
            self._file.strip()

    def write(self, string):
        """Appends the input string to the current file."""
        self._file += str(string)
        self.save()

    def overwrite(self, string):
        """Overwrites the data in the file to the input string."""
        self._file = str(string)
        self.save()

    def read(self):
        """Return the contents of the file."""
        return self._file

    def save(self):
        """Save what we've done to the file to the actual file"""
        with open(self._file_name, "w", encoding='utf-8') as file:
            file.write(self._file)

    def __str__(self):
        """Returns a str of the contents of the file"""
        return self._file
