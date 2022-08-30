import os
import pandas as pd
from skimage import io
import numpy as np
import numpy.typing
from typing import List, Tuple
import cv2


DEFAULT_MODEL_INPUT_SHAPE = (28, 28)


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

#     img = cv2.imread(path_to_image, cv2.IMREAD_GRAYSCALE)
    img = io.imread(path_to_image, as_gray=True)
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


# Filter Fonts by Name
NOT_NEEDED = {
    "adamiani",
    "agremyn",
    "amerabgec",
    "ashesha",
    "bombei",
    "constitution",
    "dara1981",
    "frapu",
    "_shirim",
    "geo-zhorzh",
    "geopixel",
    "geoalami",
    "geo_bomb",
    "geo_chveu",
    "geo_dabali",
    "geo_devi",
    "geo_doch",
    "geo_george",
    "geo_gordeladzde",
    "geo_graniti",
    "geo_kalami",
    "geo_kiknadze",
    "geo_kvamli",
    "geo_lado_",
    "geo_lortki",
    "geo_maghali",
    "geo_mdzimiseburi",
    "geo_mrude",
    "geo_mziur",
    "geo_nana",
    "geo_orqidea",
    "geo_pakizi",
    "geo_phunji",
    "geo_picasso",
    "geo_salkhino",
    "geo_shesha",
    "geo_shirim",
    "geo_times",
    "geo_veziri",
    "geo_vicro",
    "geo_victoria",
    "geo_zghapari",
    "_satellite",
    "goturi",
    "gugeshashvili",
    "_kaxa-deko",
    "_kvadro",
    "misha.nd.t",
    "misha_nd-",
    "muqara",
    "phunji_mtavruli",
    "tablon_regular",
    "talguri_rs",
    "teo_heavy",
    "ucnobi",
    "vehsapi-regular",
    "xshevardnadze",
}


def is_good_font(font_name: str) -> bool:
    return not any([f in font_name for f in NOT_NEEDED])


ORIGINAL_DATA_PATH = os.path.join("data", "raw", "alphabet")

ALPHABET_CLASSES = {}
for alph in os.listdir(ORIGINAL_DATA_PATH):
    ALPHABET_CLASSES[alph] = [
        os.path.join(alph, f)
        for f in os.listdir(os.path.join(ORIGINAL_DATA_PATH, alph))
        if is_good_font(os.path.join(alph, f))
    ]


data = pd.DataFrame(ALPHABET_CLASSES.items(), columns=["LABEL", "PATH"])
data = data.explode("PATH", ignore_index=True)
data["PATH"] = data["PATH"].apply(lambda x: os.path.join(ORIGINAL_DATA_PATH, x))
data['DATA'] = data['PATH'].apply(lambda x: load_image(x))
data['DATA_PREPROCESSED'] = data['DATA'].apply(lambda x: remove_extra_space_around_characters(preprocess_image(x), extra_space_value=0))
data['DATA_PREPROCESSED'] = data['DATA_PREPROCESSED'].apply(lambda x: zero_padding(x))
data = data[data['DATA_PREPROCESSED'].apply(lambda x: x.shape==(28,28))]

###

data = data.reset_index()

cnn_data = np.zeros(shape=(data['DATA_PREPROCESSED'].shape[0], 28, 28))
for i in range(cnn_data.shape[0]):
    cnn_data[i] = data['DATA_PREPROCESSED'][i]
cnn_data = cnn_data.astype('float32')    


### 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

le = LabelEncoder()
labels = le.fit_transform(data['LABEL'])

x_train_cnn, x_test_cnn, y_train_cnn, y_test_cnn = train_test_split(cnn_data, labels, test_size=0.1, random_state=4)
x_train_cnn, x_val_cnn, y_train_cnn, y_val_cnn = train_test_split(x_train_cnn, y_train_cnn, test_size=0.1, random_state=4)

# one-hot encode target column

y_train = to_categorical(y_train_cnn)
y_test = to_categorical(y_test_cnn)
y_val = to_categorical(y_val_cnn)

x_train_cnn = x_train_cnn.reshape(x_train_cnn.shape[0], 28, 28, 1)
x_test_cnn = x_test_cnn.reshape(x_test_cnn.shape[0], 28, 28, 1)
x_val_cnn = x_val_cnn.reshape(x_val_cnn.shape[0], 28, 28, 1)


### Model imports
from tensorflow import keras
from tensorflow.keras.models import Sequential,save_model,load_model
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from keras.layers import Dense, Activation, Flatten
from keras.preprocessing.image import ImageDataGenerator


### Model Definition

model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(33, activation='softmax'))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# Train model
history = model.fit(x_train_cnn, y_train, validation_data=(x_val_cnn, y_val), epochs=15)

# Save model
model.save('model_latest.h5')
np.save('label_encoder_latest.npy', le.classes_)