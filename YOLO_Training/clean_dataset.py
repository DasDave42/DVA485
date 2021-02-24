file = []
with open('train.txt') as fups:
    file = fups.readlines()

for line in file:
    infos = line.split(" ")
    if len(infos) == 1:
        print("too short! line:" + str(file.index(line) + 1))
    else:
        for item in infos:
            box = item.split(",")
            if len(box) > 1:
                if len(box) > 5:
                    print("box too long line:" + str(file.index(line) + 1))
                if str(box[- 1]) not in ["1", "0", "1\n", "0\n"]:
                    print("last item wrong: " + str(box[- 1]) + " line:" + str(file.index(line) + 1))