from typing import List


def input_for_frontend(corners: List[List[int]], json_path: str) -> None:
    '''
    function generates json on the given path for frontend usage, each corner having its id as string.
    example: {"id_0": [[0, 1],[0, 3],[2, 0],[1, 0]]}

    Parameters
    -----
    corners: list of list of ints
    json_path: str

    Returns
    -----
    None
    '''
    import json
    id_corners = {str(i_d): corners[i_d] for i_d in range(len(corners))}
    with open(json_path, "w") as f:
        json.dump(id_corners, f)


def output_from_frontend() -> None:
    pass