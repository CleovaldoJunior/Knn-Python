f = open("Skin_NonSkin.txt")
skin = open("Skin.data","w+")
for line in f.readlines():
    skin.write(line.replace("	",','))
f.close()
skin.close()