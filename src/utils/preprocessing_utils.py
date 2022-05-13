import os
from typing import Any, List, Tuple

import cv2
import numpy as np
import numpy.typing


def load_image(path_to_image: str) -> np.typing.NDArray[np.uint8]:
    """
    function loads image as grayscale and returns it as numpy array

    Parameters
    -----
    path_to_image: str

    Returns
    -----
    np.typing.NDArray[np.uint8]
    """
    assert os.path.exists(path_to_image)

    img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    return np.array(img, dtype=np.uint8)


def preprocess_image(
    grayscale_image: np.typing.NDArray[np.uint8],
) -> np.typing.NDArray[np.uint8]:
    """
    function takes grayscale image as array returns image as binary array

    Parameters
    -----
    grayscale_image: np.typing.NDArray[np.uint8]

    Returns
    -----
    np.typing.NDArray[np.uint8]
    """

    blur = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    th3 = cv2.adaptiveThreshold(
        grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    th3 = cv2.bitwise_not(th3)

    return np.array(th3, dtype=np.uint8)


def get_bounding_boxes(
    binary_image: np.typing.NDArray[np.uint8],
) -> List[List[Tuple[Any, ...]]]:
    """
    function takes binary image as array returns list of bounding boxes around possible characters

    Parameters
    -----
    binary_image: np.typing.NDArray[np.uint8]

    Returns
    -----
    List[List[Tuple[Any, ...]]]
    """
    bounding_boxes = []
    contours, hierarchy = cv2.findContours(
        binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    # find the rectangle around each contour
    for num in range(0, len(contours)):
        # make sure contour is for letter and not cavity
        if hierarchy[0][num][3] == -1:
            left = tuple(contours[num][contours[num][:, :, 0].argmin()][0])
            right = tuple(contours[num][contours[num][:, :, 0].argmax()][0])
            top = tuple(contours[num][contours[num][:, :, 1].argmin()][0])
            bottom = tuple(contours[num][contours[num][:, :, 1].argmax()][0])
            bounding_boxes.append([top, right, bottom, left])
    return bounding_boxes


def get_corners(bounding_boxes: List[List[Tuple[int, int]]]) -> List[List[List[int]]]:
    """
    function takes bounding_boxes and returns corners of bounding boxes

    Parameters
    -----
    bounding_boxes:  List[List[Tuple[int, int]]]

    Returns
    -----
    List[List[List[int]]]
    """

    def find_corners(bounding_box: List[Tuple[int, int]]) -> List[List[int]]:
        """
        function finds and returns the corners of the single box given the top, bottom, left, and right maximum pixels

        Parameters
        -----
        bounding_box:  List[Tuple[int, int]]

        Returns
        -----
        List[List[int]]
        """

        c1 = [int(bounding_box[3][0]), int(bounding_box[0][1])]
        c2 = [int(bounding_box[1][0]), int(bounding_box[0][1])]
        c3 = [int(bounding_box[1][0]), int(bounding_box[2][1])]
        c4 = [int(bounding_box[3][0]), int(bounding_box[2][1])]
        return [c1, c2, c3, c4]

    corners = []
    # find the edges of each bounding box
    for bx in bounding_boxes:
        corners.append(find_corners(bx))
    return corners


def get_areas(boxes_corners: List[List[List[int]]]) -> List[int]:
    """
    function calculates and returns areas of each box given list of bounding boxes corners

    Parameters
    -----
    boxes_corners:  List[List[List[int]]]

    Returns
    -----
    List[int]
    """

    def find_area(box_corners: List[List[int]]) -> int:
        """
        function calculates and returns areas given box corners coordinates

        Parameters
        -----
        box_corners:  List[List[int]]

        Returns
        -----
        int
        """
        return abs(box_corners[0][0] - box_corners[1][0]) * abs(
            box_corners[0][1] - box_corners[3][1]
        )

    areas = []
    # go through each corner and append its areas to the list
    for corner in boxes_corners:
        areas.append(find_area(corner))
    return areas


def filter_by_area(
    areas: List[int], boxes_corners: List[List[List[int]]]
) -> Tuple[List[int], List[List[List[int]]]]:
    """
    function filters areas and boxes corners by mean value

    Parameters
    -----
    areas: List[int]
    boxes_corners: List[List[List[int]]]

    Returns
    -----
    Tuple[List[int], List[List[List[int]]]]
    """
    assert len(areas) == len(boxes_corners)

    areas_np: np.typing.NDArray[np.uint16] = np.asarray(
        areas, dtype=np.uint16
    )  # organize list into array format
    boxes_corners_np: np.typing.NDArray[np.object_] = np.array(
        boxes_corners, dtype=np.object_
    )

    mask = np.where(areas_np > 0)
    non_zero_areas = areas_np[mask]
    non_zero_corners = boxes_corners_np[mask]

    avg_area = np.mean(areas_np)  # find average area
    std_area = np.std(areas_np)  # find standard deviation of area

    mask = np.where(
        (non_zero_areas > (np.mean(non_zero_areas) - np.mean(non_zero_areas) / 2))
    )
    mean_areas = non_zero_areas[mask]
    mean_corners = non_zero_corners[mask]
    return mean_areas.tolist(), mean_corners.tolist()
