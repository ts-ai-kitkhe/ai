import os
import pandas as pd


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
