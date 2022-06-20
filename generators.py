from group_generator import generate_all_selections as PS

def lol(items: list)->list:
	return [[item] for item in items]

def gen():
	yield lol([1,2])
	yield lol([3,4])
	yield lol([5])

for item in PS(gen()):
	print(item)

