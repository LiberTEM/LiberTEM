from libertem import masks


def get_roi(params, shape):

    if "roi" not in params or ("shape" not in params["roi"]):
        return None
    params = params["roi"]
    ny, nx = tuple(shape)
    if params["shape"] == "disk":
        roi = masks.circular(
            params["cx"],
            params["cy"],
            nx, ny,
            params["r"],
        )
    elif params["shape"] == "rect":
        roi = masks.rectangular(
            params["x"],
            params["y"],
            params["width"],
            params["height"],
            nx, ny,
        )
    else:
        raise NotImplementedError("unknown shape %s" % params["shape"])
    return roi
