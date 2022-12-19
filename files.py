import json


def setDataFile(n, m, p, w1, w2):
    with open('info.json', 'w', encoding="utf-8") as w:
        temp_dict = {"n": n, "m": m, "neurons": p, "w1": w1, "w2": w2}
        json.dump(temp_dict, w, indent=2)


def getDataFile():
    with open('info_2.json', 'r', encoding="utf-8") as file:
        info = json.load(file)
        return info['n'], info['m'], info['neurons'], info['w1'], info['w2']
