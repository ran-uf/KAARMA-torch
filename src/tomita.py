import numpy as np


def generate_tomita_sequence(num, length, tp=1):
    if tp == 4:
        string = np.zeros((num, length))
        label = np.zeros((num, length))
        for i in range(num):
            while label[i, int(length * 0.5)] == 0:
                string[i, :] = (np.random.random(length) > 0.5).astype(np.float32)
                for j in range(length):
                    label[i, j] = check_label(string[i, :j + 1], tp)
        return string, label
    else:
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


def train_data(grammars, batch_size, num=2048):
    t_x = []
    t_y = []
    for _ in range(int(num / batch_size)):
        for g in grammars:
            lgh = np.random.randint(10, 80)
            _x, _y = generate_tomita_sequence(batch_size, lgh, g)
            # string_x, string_y = generate_tomita_sequence(options.batch_size, length, j + 4)
            # t_x.append(torch.from_numpy(_x.astype(np.float32)).to(dev))
            # t_y.append(torch.from_numpy(_y.astype(np.float32)).to(dev))
            t_x.append(_x.astype(np.float32))
            t_y.append(_y.astype(np.float32))
    return t_x, t_y


def train_data_seq(grammars, batch_size, num=2048):
    t_x = []
    t_y = []
    for i in [0, 1]:
        for _ in range(int(num / batch_size)):
            idx_g = np.random.permutation(grammars[:i + 1])
            xx = []
            yy = []
            for g in idx_g:
                lgh = np.random.randint(50, 100)
                _x, _y = generate_tomita_sequence(batch_size, lgh, g)
                # string_x, string_y = generate_tomita_sequence(options.batch_size, length, j + 4)
                xx.append(_x)
                yy.append(_y)
            xx = np.hstack(xx)
            yy = np.hstack(yy)
            t_x.append(xx.astype(np.float32))
            t_y.append(yy.astype(np.float32))

    return t_x, t_y


def test_data(grammars):
    t_x = []
    t_y = []
    for g in grammars:
        _x, _y = generate_tomita_sequence(64, 128, g)
        t_x.append(_x)
        t_y.append(_y)
    t_x = np.vstack(t_x)
    t_y = np.vstack(t_y)
    t_x = t_x.astype(np.float32)
    t_y = t_y.astype(np.float32)
    return t_x, t_y


def test_data_seq(grammars):
    def per(_l):
        r = [np.random.choice(_l, 1)]
        for _ in range(3):
            a = np.random.choice(_l, 1)
            while a == r[-1]:
                a = np.random.choice(_l, 1)
            r.append(a)
        return r
    __x = []
    __y = []
    for _ in range(100):
        t_x = []
        t_y = []
        for g in per(grammars):
            _x, _y = generate_tomita_sequence(1, 60, g)
            t_x.append(_x)
            t_y.append(_y)
        t_x = np.hstack(t_x)
        t_y = np.hstack(t_y)
        __x.append(t_x)
        __y.append(t_y)

    _x = np.vstack(__x)
    _y = np.vstack(__y)
    t_x = _x.astype(np.float32)
    t_y = _y.astype(np.float32)
    return t_x, t_y


if __name__ == '__main__':
    strings = generate_tomita(50, 3, 7)
    print('done')
