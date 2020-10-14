import os
import glob
import argparse

""" Used to quickly remove result images created with names not containing '000' (i.e. not divisible by 1000) """

parser = argparse.ArgumentParser(description='Process some directory.')
parser.add_argument('-dir', metavar='N', type=str, help='the dir to be cleaned in')
args = parser.parse_args()

os.chdir(args.dir)
for file in glob.glob("*.png"):
	if not "000" in file:
		print("would delete " + args.dir + file)
		
text = input("Press y to confirm or n to cancel\n")

if text == "y":
	for file in glob.glob("*.png"):
		if not "000" in file:
			print("deleting " + args.dir + file)
			os.remove(file)
elif text == "n":
	print("ok, not deleting these files")