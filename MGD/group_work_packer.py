#!/usr/bin/env python3
#
# Author: Ludovic Hofer
#
# This program is a small tool that allow to create simple uniform archive for
# groups of students

import argparse
import csv
import os
import tarfile
import sys

class Student:
    def __init__(self, last_name="LastName", first_name = "FirstName"):
        self.last_name = last_name.upper()
        self.first_name = first_name.title()

    def __repr__(self):
        return "[" + self.last_name + "," + self.first_name + "]"

class Group:
    # The maximal number of students in a group before
    max_explicit_size = 3

    # The separator between LastName and FirstName
    name_separator = '.'

    # The separator between GroupMembers
    student_separator = '_'

    def __init__(self, students = []):
        # Sorting student based on last_name first and then first_name
        # Note: Still some unexpected behavior possible if last_name of a
        #       student is prefix of another student last_name
        self.students = sorted(students, key= lambda s : s.last_name + "|" + s.first_name)

    def getKey(self):
        """
        Return the group name based on students names
        """
        use_full_names = len(self.students) <= self.max_explicit_size
        key = ""
        for s in self.students:
            if len(key) != 0:
                key += self.student_separator
            if use_full_names:
                key += s.last_name + self.name_separator + s.first_name
            else:
                key += s.last_name[0] + s.first_name[0]
        return key

    def __repr__(self):
        return str(self.students)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("project_path", help="The path to the project directory")
    args = parser.parse_args()

    if not os.path.isdir(args.project_path):
        raise RuntimeError("'"+ args.project_path + "' is not a valid directory path")
    group_path = os.path.join(args.project_path, "group.csv")
    if not os.path.isfile(group_path):
        raise RuntimeError("No file named 'group.csv' in " + args.project_path)
    group = None
    with open(group_path) as f:
        students = []
        reader = csv.DictReader(f, fieldnames=["LastName","FirstName"])
        for row in reader:
            students.append(Student(row["LastName"],row["FirstName"]))
        group = Group(students)
    files_path = os.path.join(args.project_path, "to_pack.txt")
    if not os.path.isfile(group_path):
        raise RuntimeError("No file named 'to_pack.txt' in " + args.project_path)
    dst_file = group.getKey() + ".tar.gz"
    archive = tarfile.open(dst_file,"w:gz")
    with open(files_path) as f:
        for line in f.readlines():
            full_path = os.path.join(args.project_path,line.strip())
            if not os.path.isfile(full_path):
                raise RuntimeError("Failed to find file" + full_path)
            # Making sure all elements are in an archive
            archive_name = os.path.join(group.getKey(),line.strip())
            archive.add(full_path, arcname= archive_name)
    archive.close()
