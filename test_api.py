import json, urllib.request

def translate(text):
    req = urllib.request.Request(
        "http://127.0.0.1:8000/translate",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read().decode("utf-8"))["malayalam"]

for s in ["I will decide tomorrow.", "Where is the nearest hospital?",
          "Please close the window."]:
    print(s, "->", translate(s))