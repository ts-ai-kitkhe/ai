from typing import List, Tuple


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


def output_from_frontend() -> None:
    pass


def get_model_response(predictions, filtered_corners):
    model_response = [
        {
            "id": i,
            "letter": predictions[i][0],
            "confidence": float(predictions[i][1]),
            "corners": filtered_corners[i],
            "top_letters": predictions[i][2][0],
            "top_confidences": [float(p) for p in predictions[i][2][1]],
        }
        for i in range(len(filtered_corners))
    ] 
    return model_response