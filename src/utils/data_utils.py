from typing import List

import numpy as np
import numpy.typing


def get_characters(
    binary_image: np.typing.NDArray[np.uint8], corners: List[List[List[int]]]
) -> List[np.typing.NDArray[np.uint8]]:
    """
    function returns list of characters images as arrays

    Parameters
    -----
    binary_image: np.typing.NDArray[np.uint8]
    corners: List[List[List[int]]]

    Returns
    -----
    List[np.typing.NDArray[np.uint8]]
    """
    corners_np: np.typing.NDArray[np.object_] = np.array(corners)
    characters = []
    for i in range(len(corners_np)):
        characters.append(
            binary_image[
                min(corners_np[i][:, [1]])[0] : max(corners_np[i][:, [1]])[0],
                min(corners_np[i][:, [0]])[0] : max(corners_np[i][:, [0]])[0],
            ]
        )
    return characters
