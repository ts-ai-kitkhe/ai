import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import numpy.typing
from skimage import io

import os
from tensorflow import keras
import numpy as np
from skimage.transform import resize
from skimage import io, color
from sklearn.preprocessing import LabelEncoder

GEO_ALPHABET: Dict[str, int] = {
    "ა": 1,
    "ბ": 2,
    "გ": 3,
    "დ": 4,
    "ე": 5,
    "ვ": 6,
    "ზ": 7,
    "თ": 8,
    "ი": 9,
    "კ": 10,
    "ლ": 11,
    "მ": 12,
    "ნ": 13,
    "ო": 14,
    "პ": 15,
    "ჟ": 16,
    "რ": 17,
    "ს": 18,
    "ტ": 19,
    "უ": 20,
    "ფ": 21,
    "ქ": 22,
    "ღ": 23,
    "ყ": 24,
    "შ": 25,
    "ჩ": 26,
    "ც": 27,
    "ძ": 28,
    "წ": 29,
    "ჭ": 30,
    "ხ": 31,
    "ჯ": 32,
    "ჰ": 33,
}

DEFAULT_MODEL_INPUT_SHAPE: Tuple[int, int] = (28, 28)


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
    # img = io.imread(path_to_image, as_gray=True)
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
    areas: List[int], boxes_corners: List[List[List[int]]], filter_value: int = 0
) -> Tuple[List[int], List[List[List[int]]], List[int]]:
    """
    function filters areas and boxes corners by filter value

    Parameters
    -----
    areas: List[int]
    boxes_corners: List[List[List[int]]]
    filter_value: int=0

    Returns
    -----
    Tuple[List[int], List[List[List[int]]], List[int]]
    """
    assert len(areas) == len(boxes_corners)

    areas_np: np.typing.NDArray[np.uint16] = np.asarray(
        areas, dtype=np.uint16
    )  # organize list into array format
    boxes_corners_np: np.typing.NDArray[np.object_] = np.array(
        boxes_corners, dtype=np.object_
    )

    mask = np.where(areas_np > filter_value)
    filtered_areas = areas_np[mask]
    filtered_corners = boxes_corners_np[mask]

    return filtered_areas.tolist(), filtered_corners.tolist(), mask[0].tolist()


def get_boxes_sides_length(
    boxes_corners: List[List[List[int]]],
) -> Tuple[List[int], List[int]]:
    """
    function takes boxes corners and returns widths and heights of boxes

    Parameters
    -----
    boxes_corners: List[List[List[int]]]

    Returns
    -----
    Tuple[List[int], List[int]]
    """
    widths: List[int] = []
    heights: List[int] = []
    for box_corners in boxes_corners:
        widths.append(abs(box_corners[0][0] - box_corners[1][0]))
        heights.append(abs(box_corners[0][1] - box_corners[3][1]))
    return widths, heights


def filter_by_sides(
    corners: List[List[List[int]]], widths: List[int], heights: List[int]
) -> Tuple[List[List[List[int]]], List[int]]:
    """
    function filters corners of boxes by sides characteristics

    Parameters
    -----
    corners: List[List[List[int]]]
    widths: List[int], heights: List[int]
    heights: List[int]

    Returns
    -----
    Tuple[List[List[List[int]]], List[int]]
    """
    mask = np.where(
        (np.array(widths) > int(np.mean(widths)))
        & (np.array(heights) > int(np.mean(heights)))
    )
    filtered_corners = np.array(corners)[mask]
    return filtered_corners.tolist(), mask[0].tolist()


def input_for_frontend(
    corners: List[List[int]], predictions: List[Tuple[str, float]], json_path: str
) -> None:
    """
    function generates json on the given path for frontend usage, each corner having its id as string.
    example: [{"id": 0, "letter": "ა", "confidence": 0.99, "corners": [[0, 1],[0, 3],[2, 0],[1, 0]]}]

    Parameters
    -----
    corners: list of list of ints
    predictions: list of Tuples of str and float
    json_path: str

    Returns
    -----
    None
    """
    import json

    assert len(corners) == len(predictions)
    model_response = [
        {
            "id": i,
            "letter": predictions[i][0],
            "confidence": float(predictions[i][1]),
            "corners": corners[i],
        }
        for i in range(len(corners))
    ]

    with open(json_path, "w", encoding="utf8") as f:
        json.dump(model_response, f, ensure_ascii=False)


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


def main():
    # %%timeit
    image_file = "data/raw/real_1.jpg"

    img = load_image(image_file)
    binary_image = preprocess_image(img)
    # characteristics
    bounding_boxes = get_bounding_boxes(binary_image)
    corners = get_corners(bounding_boxes)
    areas = get_areas(corners)
    widths, heights = get_boxes_sides_length(corners)

    # filters
    filtered_areas, filtered_corners, area_mask = filter_by_area(
        areas, corners, filter_value=0
    )
    filtered_widths, filtered_heights = (
        np.array(widths)[np.array(area_mask)].tolist(),
        np.array(heights)[np.array(area_mask)].tolist(),
    )

    filtered_corners, side_mask = filter_by_sides(
        filtered_corners, filtered_widths, filtered_heights
    )
    filtered_widths, filtered_heights = (
        np.array(filtered_widths)[np.array(side_mask)].tolist(),
        np.array(filtered_heights)[np.array(side_mask)].tolist(),
    )
    filtered_areas = np.array(filtered_areas)[np.array(side_mask)].tolist()

    filtered_areas, filtered_corners, area_mask = filter_by_area(
        filtered_areas, filtered_corners, filter_value=np.mean(filtered_areas) / 2
    )
    filtered_widths, filtered_heights = (
        np.array(widths)[np.array(area_mask)].tolist(),
        np.array(heights)[np.array(area_mask)].tolist(),
    )

    characters = get_characters(binary_image, filtered_corners)

    model_name = "models/first_cnn/first_cnn.h5"
    label_encoder_name = "models/first_cnn/first_cnn_label_encoder.npy"

    model = keras.models.load_model(model_name)
    le = LabelEncoder()
    le.classes_ = np.load(label_encoder_name, allow_pickle=True)
    le_name_mapping = dict(zip(le.transform(le.classes_), le.classes_))

    predictions = []
    for c in characters:
        preprocessed = zero_padding(
            remove_extra_space_around_characters(c, extra_space_value=0)
        )
        image_cnn = np.zeros(shape=(1, 28, 28))
        image_cnn[0] = zero_padding(
            remove_extra_space_around_characters(c, extra_space_value=0)
        )
        prediction = model.predict(image_cnn)
        letter, confidence = np.argmax(prediction), np.max(prediction)

        predictions.append((le_name_mapping[letter], confidence))

    input_for_frontend(filtered_corners, predictions, "models_predictions.json")


if __name__ == "__main__":
    main()
