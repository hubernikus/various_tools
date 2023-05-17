def hex_to_rgba(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def hex_to_rgba_float(value):
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) / 256.0 for i in range(0, lv, lv // 3))


def rgb_to_hex(rgb):
    return "#%02x%02x%02x" % rgb
