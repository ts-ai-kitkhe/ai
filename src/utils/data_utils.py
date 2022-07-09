from typing import List, Tuple

import cv2
import numpy as np
import numpy.typing

from src.utils.constant_utils import DEFAULT_MODEL_INPUT_SHAPE


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


def zero_padding(
    binary_image: np.typing.NDArray[np.uint8],
    desired_shape: Tuple[int, int] = DEFAULT_MODEL_INPUT_SHAPE,
    pad_value: int = 0,
) -> np.typing.NDArray[np.uint8]:
    """
    function pads given image with desired padding value to desired shape

    Parameters
    -----
    binary_image: np.typing.NDArray[np.uint8]
    desired_shape: Tuple[int, int]
    pad_value: int

    Returns
    -----
    np.typing.NDArray[np.uint8]
    """
    # if any side > desired shape -> scaling
    scale = 1
    if (
        binary_image.shape[0] > desired_shape[0]
        and binary_image.shape[1] > desired_shape[1]
    ):
        if binary_image.shape[0] > binary_image.shape[1]:
            scale = desired_shape[0] / binary_image.shape[0]
        else:
            scale = desired_shape[1] / binary_image.shape[1]
    elif binary_image.shape[0] > desired_shape[0]:
        scale = desired_shape[0] / binary_image.shape[0]

    elif binary_image.shape[1] > desired_shape[1]:
        scale = desired_shape[1] / binary_image.shape[1]

    width = int(binary_image.shape[1] * scale)
    height = int(binary_image.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(binary_image, dim)
    pad = np.full(desired_shape, np.uint8(pad_value))
    # if curr shape < desired shape -> zero padding
    if (
        resized.shape[0] <= desired_shape[0]
        and binary_image.shape[1] <= desired_shape[1]
    ):
        l = (pad.shape[0] - resized.shape[0]) // 2
        u = (pad.shape[1] - resized.shape[1]) // 2
        pad[l : resized.shape[0] + l, u : resized.shape[1] + u] = resized
    else:
        pad[
            : min(desired_shape[0], binary_image.shape[0]),
            : min(desired_shape[1], binary_image.shape[1]),
        ] = binary_image[: desired_shape[0], : desired_shape[1]]
    return pad


def remove_extra_space_around_characters(
    binary_image: np.typing.NDArray[np.uint8], extra_space_value: int = 255
) -> np.typing.NDArray[np.uint8]:
    """
    function removes extra space filled with some arbitrary value around characters in binary image

    Parameters
    -----
    binary_image: np.typing.NDArray[np.uint8]
    extra_space_value: int

    Returns
    -----
    np.typing.NDArray[np.uint8]
    """

    n_rows, n_cols = binary_image.shape
    upper_row, lower_row, left_col, right_col = 0, n_rows, 0, n_cols

    for i in range(n_rows):
        if sum(binary_image[i, :]) == n_cols * extra_space_value:
            upper_row = i
        else:
            break

    for i in range(n_rows):
        if sum(binary_image[n_rows - i - 1, :]) == n_cols * extra_space_value:
            lower_row = n_rows - i - 1
        else:
            break

    for j in range(n_cols):
        if sum(binary_image[:, j]) == n_rows * extra_space_value:
            left_col = j
        else:
            break

    for j in range(n_cols):
        if sum(binary_image[:, n_cols - j - 1]) == n_rows * extra_space_value:
            right_col = n_cols - j - 1
        else:
            break
    removed_extra_space_image = binary_image[
        upper_row : lower_row + 1, left_col : right_col + 1
    ]
    if (
        removed_extra_space_image.shape[0] == 0
        or removed_extra_space_image.shape[1] == 1
    ):
        return binary_image
    return removed_extra_space_image
