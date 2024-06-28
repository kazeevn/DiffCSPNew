import json
from pyxtal.symmetry import Group
from pyxtal import pyxtal


with open("WyckoffTransformer_mp_20.json") as f:
    data = json.load(f)

print(data[0])
g = Group(data[0]['group'])
print(g)

c = pyxtal()
c.from_random(**data[0])
print(c)
