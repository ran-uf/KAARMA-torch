import numpy as np


def generate_tomita_sequence(num, length, tp=1):
    string = np.zeros((num, length))
    label = np.zeros((num, length))
    for i in range(num):
        string[i, :] = (np.random.random(length) > 0.5).astype(np.float32)
        for j in range(length):
            label[i, j] = check_label(string[i, :j + 1], tp)
    return string, label


def generate_tomita(num, length, tp=1):
    string = []
    label = []
    for i in range(0, num, 2):
        string.append(tomita_string(length, 1, tp))
        label.append(1)
        if string[-1] is None:
            del string[-1]
            del label[-1]

        string.append(tomita_string(length, 0, tp))
        label.append(0)
        if string[-1] is None:
            del string[-1]
            del label[-1]
    return np.array(string), np.array(label)


def tomita_string(length, label, tp):
    for i in range(100):
        string = np.random.random(length) > 0.5
        string = string.astype(np.float32)
        if check_label(string, tp) == label:
            return string
    return None


def check_label(string, tp):
    if tp == 1:
        for a in string:
            if a == 0:
                return 0
        return 1
    elif tp == 2:
        for (a, b) in zip(string[:-1], string[1:]):
            if a == b:
                return 0
        return 1
    elif tp == 3:
        num_0, num_1 = 0, 0
        for a in string:
            if a == 1:
                if num_0 == 0:
                    num_1 += 1
                elif num_0 % 2 == 1:
                    return 0
            else:
                if num_1 % 2 == 1:
                    num_0 += 1
                else:
                    num_0, num_1 = 0, 0
        if num_0 % 2 == 1 and num_1 % 2 == 1:
            return 0
        return 1
    elif tp == 4:
        num_0 = 0
        for a in string:
            if a == 0:
                num_0 = num_0 + 1
                if num_0 >= 3:
                    return 0
            elif a == 1:
                num_0 = 0
        return 1
    elif tp == 5:
        num_0, num_1 = 0, 0
        for a in string:
            if a == 0:
                num_0 += 1
            elif a == 1:
                num_1 += 1
        if num_0 % 2 == 0 and num_1 % 2 == 0:
            return 1
        else:
            return 0
    elif tp == 6:
        num_0, num_1 = 0, 0
        for a in string:
            if a == 0:
                num_0 += 1
            elif a == 1:
                num_1 += 1
        if (num_0 - num_1) % 3 == 0:
            return 1
        else:
            return 0
    elif tp == 7:
        state = 0
        last = 0
        for a in string:
            if a != last:
                state += 1
                if state > 3:
                    return 0
        return 1


if __name__ == '__main__':
    strings = generate_tomita(50, 3, 7)
    print('done')
