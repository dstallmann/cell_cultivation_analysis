from PIL import Image
import glob, os

'''Converts images from 4 channel RGBA to 1 channel luminosity'''

os.chdir("../data/128p_pc/train/")
names = []
loaded_images = []
for file in glob.glob("*.jpg"):#("*.png"):
	img = Image.open(file)
	img2 = img.convert('L')
	names.append(file)
	loaded_images.append(img2)
	print(file + " done")

for i in range(0,len(loaded_images)):
	print(str(i) + " done")
	loaded_images[i].save("done/" + names[i],"PNG")