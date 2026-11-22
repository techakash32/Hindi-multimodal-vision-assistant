# hindi_text.py
# Offline Hindi phrases, so safety alerts and status messages never
# depend on the network.

CLASS_HI = {
    "person": "व्यक्ति", "bicycle": "साइकिल", "car": "कार",
    "motorcycle": "मोटरसाइकिल", "airplane": "हवाई जहाज़", "bus": "बस",
    "train": "ट्रेन", "truck": "ट्रक", "boat": "नाव",
    "traffic light": "ट्रैफ़िक लाइट", "fire hydrant": "फ़ायर हाइड्रेंट",
    "stop sign": "स्टॉप साइन", "parking meter": "पार्किंग मीटर",
    "bench": "बेंच", "bird": "पक्षी", "cat": "बिल्ली", "dog": "कुत्ता",
    "horse": "घोड़ा", "sheep": "भेड़", "cow": "गाय", "elephant": "हाथी",
    "bear": "भालू", "zebra": "ज़ेब्रा", "giraffe": "जिराफ़",
    "backpack": "बैग", "umbrella": "छाता", "handbag": "हैंडबैग",
    "tie": "टाई", "suitcase": "सूटकेस", "frisbee": "फ्रिस्बी",
    "skis": "स्की", "snowboard": "स्नोबोर्ड", "sports ball": "गेंद",
    "kite": "पतंग", "baseball bat": "बल्ला", "baseball glove": "दस्ताना",
    "skateboard": "स्केटबोर्ड", "surfboard": "सर्फ़बोर्ड",
    "tennis racket": "रैकेट", "bottle": "बोतल", "wine glass": "वाइन ग्लास",
    "cup": "कप", "fork": "कांटा", "knife": "चाकू", "spoon": "चम्मच",
    "bowl": "कटोरा", "banana": "केला", "apple": "सेब",
    "sandwich": "सैंडविच", "orange": "संतरा", "broccoli": "ब्रोकली",
    "carrot": "गाजर", "hot dog": "हॉट डॉग", "pizza": "पिज़्ज़ा",
    "donut": "डोनट", "cake": "केक", "chair": "कुर्सी", "couch": "सोफ़ा",
    "potted plant": "गमले का पौधा", "bed": "बिस्तर", "dining table": "मेज़",
    "table": "मेज़", "door": "दरवाज़ा", "toilet": "शौचालय", "tv": "टीवी",
    "laptop": "लैपटॉप", "mouse": "माउस", "remote": "रिमोट",
    "keyboard": "कीबोर्ड", "cell phone": "मोबाइल फ़ोन",
    "microwave": "माइक्रोवेव", "oven": "ओवन", "toaster": "टोस्टर",
    "sink": "सिंक", "refrigerator": "फ्रिज", "book": "किताब",
    "clock": "घड़ी", "vase": "फूलदान", "scissors": "कैंची",
    "teddy bear": "टेडी बियर", "hair drier": "हेयर ड्रायर",
    "toothbrush": "टूथब्रश",
}

from .strings import MESSAGES  # noqa: E402,F401  (kept here for compatibility)


def describe_hindi(obj, frame_shape):
    """Hindi version of Detector.describe(): 'व्यक्ति, बहुत पास, सामने'."""
    name, _conf, _xyxy, xc, area = obj
    _h, w = frame_shape[:2]

    direction = (
        "आपके बाईं ओर" if xc < w * 0.33
        else "आपके दाईं ओर" if xc > w * 0.66
        else "सामने"
    )

    proximity = (
        "बहुत पास" if area > 0.15
        else "पास" if area > 0.05
        else ""
    )

    parts = [CLASS_HI.get(name, name)]
    if proximity:
        parts.append(proximity)
    parts.append(direction)

    return ", ".join(parts)
