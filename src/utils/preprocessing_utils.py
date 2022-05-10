import cv2
import numpy as np



def load_image(img_path: str) -> np.ndarray:
    '''
    returns grayscaled image as numpy ndarray
    '''
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return img


def preprocess_image(img: np.ndarray)-> np.ndarray:
    '''
    return image as an array of 0's and 1's
    '''
    #apply adaptive threshold to image
    blur = cv2.GaussianBlur(img,(5,5),0)

    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    th3 = cv2.bitwise_not(th3)
    
    return th3

def get_bounding_boxes(bitwise_image: np.ndarray):
    '''
    returns list of bounding boxes
    '''
    bounding_boxes = []
    contours, heirar = cv2.findContours(bitwise_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    #find the rectangle around each contour
    for num in range(0,len(contours)):
        #make sure contour is for letter and not cavity
        if(heirar[0][num][3] == -1):
            left = tuple(contours[num][contours[num][:,:,0].argmin()][0])
            right = tuple(contours[num][contours[num][:,:,0].argmax()][0])
            top = tuple(contours[num][contours[num][:,:,1].argmin()][0])
            bottom = tuple(contours[num][contours[num][:,:,1].argmax()][0])
            bounding_boxes.append([top,right,bottom,left])
    return bounding_boxes  


def get_corners(bounding_boxes):
    '''
    returns corners of bounding boxes
    '''
    
    def find_corners(bounding_box):
        '''
        finds the corners given the top, bottom, left, and right maximum pixels
        return list of lists
        '''
        c1 = [int(bounding_box[3][0]),int(bounding_box[0][1])]
        c2 = [int(bounding_box[1][0]),int(bounding_box[0][1])]
        c3 = [int(bounding_box[1][0]),int(bounding_box[2][1])]
        c4 = [int(bounding_box[3][0]),int(bounding_box[2][1])]
        return [c1 , c2, c3, c4]

    corners = []
    #find the edges of each bounding box
    for bx in bounding_boxes:
        corners.append(find_corners(bx))
    return corners


def get_areas(corners):
    '''
    returns area of each box given list of nounding boxes corners 
    '''
    
    def findArea(c1):
        '''
        given list of box corners coordinates returns area
        '''
        return abs(c1[0][0]-c1[1][0])*abs(c1[0][1]-c1[3][1])    

    area = []
    #go through each corner and append its area to the list
    for corner in corners:
        area.append(findArea(corner))
    area = np.asarray(area) #organize list into array format
    return area


def filter_by_area(areas, corners):
    '''
    filters area and corners by mean value
    '''
    assert len(areas) == len(corners)

    mask = np.where(areas > 0)
    non_zero_areas = areas[mask]
    non_zero_corners = np.array(corners)[mask] if type(corners) == list else corners[mask]
    
    avg_area = np.mean(areas) #find average area
    std_area = np.std(areas) #find standard deviation of area

    mask = np.where((non_zero_areas > (np.mean(non_zero_areas) - np.mean(non_zero_areas)/2)))
    mean_areas = non_zero_areas[mask]
    mean_corners = non_zero_corners[mask]  
    return mean_areas, mean_corners


def get_characters(img, corners):
    '''
    returns list of characters images
    '''
    characters = []
    for i in range(len(corners)):
        characters.append(img[min(corners[i][:,[1]])[0]: max(corners[i][:,[1]])[0], min(corners[i][:,[0]])[0] : max(corners[i][:,[0]])[0]])
    return characters


def generate_json(corners, json_path):
    '''
    generates json of bounding boxes corners for front-end input
    '''
    import json
    id_corners = {str(i_d): corners[i_d] for i_d in range(len(corners))}
    with open(json_path, "w") as f:
        json.dump(id_corners, f)