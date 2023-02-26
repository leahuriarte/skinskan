import os
import shutil

# specify the directory containing the text files to be organized
source_dir = '/Users/Leah/wound/archive'


# loop through the text files in the source directory
for filename in os.listdir(source_dir):
    # check if the file is a text file
    if filename.endswith('.txt'):
        # read the contents of the file
        with open(os.path.join(source_dir, filename), 'r') as file:
            content = file.read()
        # extract the first digit from the content
        first_digit = next((int(c) for c in content if c.isdigit()), None)
        # move the file to the appropriate directory based on the first digit
        filename = filename[0:-3]
        filename += "jpg"
        if filename in os.listdir(source_dir):
            if first_digit == 0:
                shutil.move(os.path.join(source_dir, filename), os.path.join(source_dir, '0_files', filename))
            elif first_digit == 1:
                shutil.move(os.path.join(source_dir, filename), os.path.join(source_dir, '1_files', filename))
            elif first_digit == 2:
                shutil.move(os.path.join(source_dir, filename), os.path.join(source_dir, '2_files', filename))

'''filename = "watever.txt"
filename = filename[0:-3]
filename += "jpg"
print(filename)'''