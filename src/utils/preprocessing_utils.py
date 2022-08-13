import os
from typing import Any, List, Tuple

import cv2
import numpy as np
import numpy.typing
from skimage import io


# This class represents a letter object.
# A letter contains the coordinates and dimensions
# of the bounding box in the image that it belongs
# to
class Letter:
    def __init__(self,coords,dims,number):
        self.id = number
        self.x = coords[0]
        self.y = coords[1]
        #dimensions in format [height,width]
        self.dimen = dims
        self.myCoor = [self.x,self.y]
        #will hold the string value of the letter when determined
        #or -1 if no value is determined.
        self.val = ""
        #the two adjacent neighbors of the letter are saved here
        # self.right
        # self.left

    def getID(self):
        return self.id

    def getY(self):
        return self.y

    def getX(self):
        return self.x

    def getCoords(self):
        return self.myCoor

    def getHeight(self):
        return self.dimen[0]

    def getWidth(self):
        return self.dimen[1]

    def getDimension(self):
        return self.dimen

    def getValue(self):
        return self.val

    def getRight(self):
        return self.right

    def getLeft(self):
        return self.left

    def getArea(self):
        return self.dimen[0]*self.dimen[1]


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


def filter_boxes_by_models_predictions(corners, predictions, confidence_threshold=0.5):
    predictions_confidence_array = np.array([p[1] for p in predictions], np.float16)
    pred_array = np.array(predictions, dtype=np.object_)
    corners_array = np.array(corners, dtype=np.object_)
    
    mask = np.where(predictions_confidence_array > confidence_threshold)
    filtered_predictions = pred_array[mask]
    filtered_corners = corners_array[mask]
    return filtered_corners, filtered_predictions, mask


def get_all_leters(corners):
    all_letters = []
    counter = 0
    for bx in corners:
        width = abs(bx[1][0] - bx[0][0])
        height = abs(bx[3][1] - bx[0][1])
        newLetter = Letter([bx[0][0],bx[0][1]],[height,width],counter)
        all_letters.append(newLetter)
        counter += 1

    all_letters.sort(key=lambda letter: letter.getY() + letter.getHeight())
    return all_letters


def project_y_letters(all_letters):
    prjYCoords = []
    for letter in all_letters:
        prjYCoords.append(letter.getY()+letter.getHeight())
    
    return prjYCoords


def find_distances_between_coordinates(y_projection):
    coorDists = [0]
    for i in range(1, len(y_projection)):
        valCur = y_projection[i]
        valPast = y_projection[i-1]
        coorDists.append(valCur-valPast)   

    return coorDists


#function finds the minimization of the weighted within-class variance
#this algorithm is adapted from:
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
def findThresh(data):
    Binsize = 50
    #find density and bounds of histogram of data
    density,bds = np.histogram(data,bins=Binsize)
    #normalize the histogram values
    norm_dens = (density)/float(sum(density))
    #find discrete cumulative density function
    cum_dist = norm_dens.cumsum()
    #initial values to be overwritten
    fn_min = np.inf
    thresh = -1
    bounds = range(1,Binsize)
    #begin minimization routine
    for itr in range(0,Binsize):
        if(itr == Binsize-1):
            break;
        p1 = np.asarray(norm_dens[0:itr])
        p2 = np.asarray(norm_dens[itr+1:])
        q1 = cum_dist[itr]
        q2 = cum_dist[-1] - q1
        b1 = np.asarray(bounds[0:itr])
        b2 = np.asarray(bounds[itr:])
        #find means
        m1 = np.sum(p1*b1)/q1
        m2 = np.sum(p2*b2)/q2
        #find variance
        v1 = np.sum(((b1-m1)**2)*p1)/q1
        v2 = np.sum(((b2-m2)**2)*p2)/q2

        #calculate minimization function and replace values
        #if appropriate
        fn = v1*q1 + v2*q2
        if fn < fn_min:
            fn_min = fn
            thresh = itr

    return thresh, bds[thresh]


def get_lines(coord_distances, all_letters, y_letters):
    # find division in distance data
    res,bthval = findThresh(coord_distances)
    # use division to distinguish between paragraphs and sentences
    lines = [[all_letters[0]]]
    IDS = [[all_letters[0].getID()]]
    count = 0

    start = 0
    end = 0
    asd = 1.0
    meanCoord = float(sum(coord_distances))/float(len(coord_distances))
    stdCoord = np.std(coord_distances)

    medPoints = []
    for num in range(0,len(coord_distances)):
        if coord_distances[num] > meanCoord + asd*stdCoord and end == 0:
            start = num
        if coord_distances[num] > meanCoord + asd*stdCoord and start > 0:
            end = num
            medPoints.append(int(start+(end-start)/2.0))
            start = num
    medPoints.append(start)

    medPoints.insert(0,0)

    lines = []
    for num in range(0,len(medPoints)):
        lines.append(y_letters[medPoints[num]])
    
    return lines


def get_sorted_lines(lines):
    import statistics
    sorted_lines = sorted(lines)
    distances_lines = [(sorted_lines[i+1] - sorted_lines[i]) for i in range(len(sorted_lines)-1)]
    distances_mean = statistics.mean(distances_lines)
    result_lines = [0]
    for i in range(len(sorted_lines) - 1):
        if distances_lines[i] > distances_mean/2:
            result_lines.append(sorted_lines[i])
    return result_lines


def get_text_lines(result_lines, model_response):
    text_lines = []
    for i in range(len(result_lines)-1):
        y0 = result_lines[i]
        y1 = result_lines[i+1]
        
        text_lines.append([m for m in model_response if m['corners'][0][1] >= y0 and m['corners'][0][1] <=y1])
    
    return text_lines

