"""User-facing text.

* Spoken text (MESSAGES, PATH) is Devanagari Hindi with everyday English
  loanwords, because the Hindi voice reads Devanagari.
* Screen text (everything ending in _DISPLAY, plus UI, EVENTS, BUSY, ...) is
  Hinglish: Hindi in Roman letters with common English words.

Developer-facing text (logs, CLI help, README) stays in English.
"""
from __future__ import annotations

TITLE = "Drishti Sahayak"

# ---------------------------------------------------------------- spoken
MESSAGES = {
    "ready": "असिस्टेंट तैयार है।",
    "busy": "प्लीज़ रुकिए, अभी काम चल रहा है।",
    "scene_unchanged": "सीन वैसा ही है।",
    "nothing_to_compare": "अभी कोई डिस्क्रिप्शन नहीं है। पहले सीन डिस्क्राइब करवाएँ।",
    "no_change": "कोई बदलाव नहीं मिला।",
    "view_changed": "सीन बदला हुआ लगता है। नया डिस्क्रिप्शन माँगें।",
    "no_text": "कोई टेक्स्ट नहीं मिला। कैमरा पास लाकर ट्राई करें।",
    "ocr_error": "टेक्स्ट पढ़ना अभी पॉसिबल नहीं है।",
    "vlm_error": "सीन डिस्क्रिप्शन अभी पॉसिबल नहीं है।",
    "camera_error": "कैमरा काम नहीं कर रहा है।",
    "translate_error": "ट्रांसलेशन अभी पॉसिबल नहीं है।",
    "added": "नया",
    "removed": "हट गया",
    # abstention: a clarification, never a generic error
    "abstain_ocr": "पक्का नहीं हूँ — कैमरा थोड़ा पास करो?",
    "abstain_vlm": "मुझे ठीक से दिख नहीं रहा, दोबारा कोशिश करें?",
    "abstain_missing": "कैमरा उपलब्ध नहीं है।",
    "abstain_conflict": "जानकारी मिल नहीं रही, थोड़ा और बताओ।",
    "hedge": "शायद",
}
PATH = {
    "obstacle_ahead": "सामने रुकावट है",
    "unclear": "रास्ता साफ़ नहीं दिख रहा",
    "none_seen": "सामने कोई रुकावट नहीं दिखी",
}

# ---------------------------------------------------------------- screen
MESSAGES_DISPLAY = {
    "ready": "Assistant taiyaar hai.",
    "busy": "Please ruko, abhi kaam chal raha hai.",
    "scene_unchanged": "Scene waisa hi hai.",
    "nothing_to_compare": "Abhi koi description nahi hai. Pehle scene describe karwayein.",
    "no_change": "Koi badlaav nahi mila.",
    "view_changed": "Scene badla hua lagta hai. Naya description maangein.",
    "no_text": "Koi text nahi mila. Camera paas laakar try karein.",
    "ocr_error": "Text padhna abhi possible nahi hai.",
    "vlm_error": "Scene description abhi possible nahi hai.",
    "camera_error": "Camera kaam nahi kar raha hai.",
    "translate_error": "Translation abhi possible nahi hai.",
    "added": "Naya",
    "removed": "Hat gaya",
    "abstain_ocr": "Pakka nahi hoon — camera thoda paas karo?",
    "abstain_vlm": "Mujhe theek se dikh nahi raha, dobara koshish karein?",
    "abstain_missing": "Camera available nahi hai.",
    "abstain_conflict": "Jaankari mil nahi rahi, thoda aur batao.",
    "hedge": "Shayad",
}
PATH_DISPLAY = {
    "obstacle_ahead": "Saamne rukaawat hai",
    "unclear": "Raasta saaf nahi dikh raha",
    "none_seen": "Saamne koi rukaawat nahi dikhi",
}

BUSY = {"read": "Text padh raha hai", "describe": "Scene describe kar raha hai"}

EVENTS = {
    "starting": "Assistant shuru ho raha hai",
    "starting_demo": "Assistant shuru ho raha hai (demo)",
    "voice_on": "Voice commands chalu",
    "voice_off": "Voice commands available nahi",
    "muted": "Awaaz band",
    "unmuted": "Awaaz chalu",
    "nothing_to_repeat": "Dohrane ke liye abhi kuch nahi hai",
    "skipped_busy": "Command skip hua: abhi kaam chal raha hai",
    "scene_same": "Scene nahi badla; description skip hua",
    "camera_lost": "Camera band ho gaya, dobara connect ho raha hai",
    "camera_back": "Camera connect ho gaya",
    "camera_dead": "Camera kaam nahi kar raha",
    "loop_crashed": "Camera loop ruk gaya; logs dekhein",
    "ocr_failed": "Text padhna fail hua",
    "vlm_failed": "Scene description fail hua",
    "empty": "(khaali)",
}

COMMAND_NAMES = {
    "read": "Text padhna", "describe": "Scene describe", "changed": "Kya badla",
    "repeat": "Dobara bolna", "mute": "Awaaz on/off", "quit": "Band karna",
}
SOURCES = {"keyboard": "keyboard", "voice": "awaaz", "web": "web", "ui": "screen", "window": "window"}

UI = {
    "skip": "Seedhe last announcement par jaayein",
    "connecting": "Connect ho raha hai…",
    "ready": "Taiyaar",
    "no_camera": "Camera nahi mila",
    "disconnected": "Connection toot gaya",
    "offline_banner": "Assistant se connection toot gaya. Dobara try ho raha hai…",
    "demo": "Demo feed",
    "privacy_local": "Privacy: frame is device se bahar nahi jaate",
    "privacy_cloud": "Privacy: OCR aur scene ke liye frame cloud par jaata hai",
    "noscript": "Is dashboard ko JavaScript chahiye. Bina screen ke sunne ke liye terminal se `visionassist run --ui headless` chalaiye.",
    "fps": "frames/sec",
    "infer": "ms per detection",
    "uptime": "Chal raha hai",
    "alt_video": "Camera ka live view, pehchaani gayi cheezon ke box ke saath",
    "nocam": "Camera respond nahi kar raha. Check karein ki woh laga hua hai aur kisi aur app mein use nahi ho raha.",
    "zone_L": "Baayein", "zone_C": "Saamne", "zone_R": "Daayein",
    "clear": "Khaali",
    "near": "Bahut paas", "mid": "Paas", "far": "Door",
    "zones_label": "Cheezein kahan hain",
    "radar": "Aas-paas ka view",
    "radar_you": "Aap",
    "radar_empty": "Abhi kuch nahi dikh raha",
    "last_said": "Abhi kya bola",
    "spoken_label": "Awaaz mein (Hindi)",
    "waiting": "Pehli announcement ka intezaar…",
    "commands": "Commands",
    "read": "Text padhein", "describe": "Scene batayein", "changed": "Kya badla",
    "repeat": "Dobara bolein", "mute": "Awaaz band karein", "unmute": "Awaaz chalu karein",
    "activity": "Activity",
    "activity_empty": "Abhi kuch nahi. “Text padhein” ya “Scene batayein” dabakar dekhein.",
    "response": "Response time",
    "col_action": "Action", "col_runs": "Baar", "col_median": "Median", "col_slow": "Sabse slow 5%",
    "act_detect": "Alerts", "act_ocr": "Text padhna", "act_vlm": "Scene description",
    "ms": "ms",
    "system": "System",
    "h_camera": "Camera", "h_camera_ok": "Frames aa rahe hain", "h_camera_bad": "Frames nahi aa rahe",
    "h_voice": "Hindi awaaz", "h_voice_demo": "Demo mein band", "h_voice_ok": "Ready", "h_voice_bad": "Load nahi hui",
    "h_cmd": "Voice commands", "h_cmd_on": "Sun raha hai", "h_cmd_off": "Band",
    "h_ocr": "Text padhne ki service", "h_ocr_ok": "Available", "h_ocr_bad": "Errors ke baad ruki hui",
    "h_vlm": "Scene description model", "h_vlm_n": "{total} mein se {ok} available",
    "check": "(check karein)", "broken": "(kaam nahi kar raha)",
    "k_detect": "Pehchaan", "k_speech": "Bola", "k_command": "Command", "k_ocr": "Text",
    "k_vlm": "Scene", "k_system": "System", "k_error": "Problem",
    "footer": "Version {version}. Yeh aapke device par chalta hai; sirf text padhne aur scene batane mein internet lagta hai.",
}
