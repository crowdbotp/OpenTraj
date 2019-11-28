import os
from parse_utils import Scale, SDD_Parsrer

path = '../../SDD'

files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file:
            files.append(os.path.join(r, file))

print(files)

tot_Bikers = 0
tot_Pedestrians = 0
tot_Bicyclists = 0
tot_Skateboarders = 0
tot_Carts = 0
tot_Cars = 0
tot_Buss = 0
for file in files:
    parser = SDD_Parsrer()
    parser.load(file)

    Bikers = 0
    Pedestrians = 0
    Bicyclists = 0
    Skateboarders = 0
    Carts = 0
    Cars = 0
    Buss = 0
    for label in parser.label_data:
        if label == "Biker":
            Bikers += 1
        elif label == "Pedestrian":
            Pedestrians += 1
        elif "Skateboard" in label:
            Skateboarders += 1
        elif label == "Cart":
            Carts += 1
        elif label == "Car":
            Cars += 1
        elif label == "Bus":
            Buss += 1

    tot_Bikers += Bikers
    tot_Pedestrians += Pedestrians
    tot_Bicyclists += Bicyclists
    tot_Skateboarders += Skateboarders
    tot_Carts += Carts
    tot_Cars += Cars
    tot_Buss += Buss

    print(file)
    print(Bikers, Pedestrians, Bicyclists, Skateboarders, Carts, Cars, Buss)

print('final result')
print(tot_Bikers, tot_Pedestrians, tot_Bicyclists, tot_Skateboarders, tot_Carts, tot_Cars, tot_Buss)


