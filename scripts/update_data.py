"""Lấy dữ liệu mới từ realclothes.co rồi tạo lại index.html.

Cập nhật: danh sách sản phẩm (D), mô tả form cho món mới (FORM),
chỉ mục tìm bằng hình (IDX) và dòng "N sản phẩm, dữ liệu ngày ...".
Giữ nguyên mọi phần khác của trang (MATV, khuyến mãi, mẫu tin nhắn...).
"""
import base64, hashlib, html as H, io, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
from PIL import Image, ImageOps

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAGE = os.path.join(ROOT, "index.html")
MODEL_DIR = os.path.join(ROOT, ".model")
CACHE_FILE = os.path.join(ROOT, ".cache", "anh.json")
SITE = "https://realclothes.co"
UA = {"User-Agent": "Mozilla/5.0 (cap-nhat-bao-gia)"}
MODEL_URL = "https://huggingface.co/Xenova/dinov2-small/resolve/main/onnx/model_quantized.onnx"
MAX_IMG = 12          # số ảnh mỗi món đưa vào tìm bằng hình
THUMB = (330, 440)    # ảnh nhỏ hiện trên trang

TYPE_CAT = {"Áo kiểu": "top", "Sơ mi": "top", "Áo thun": "top", "Áo dệt kim": "top",
            "váy": "bot", "Quần": "bot", "Jean": "bot", "Maxi": "bot",
            "Đầm": "other", "Blazer": "other", "Áo dài": "other",
            "Phụ kiện": "acc", "Khác": "acc"}


def log(*a):
    print(*a, flush=True)


def fetch(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:
            if i == tries - 1:
                raise
            log(f"  thử lại {url}: {e}")
            time.sleep(3 * (i + 1))


# ---------- đọc web ----------
def all_products():
    out = []
    for page in range(1, 100):
        data = json.loads(fetch(f"{SITE}/collections/all/products.json?limit=50&page={page}"))
        batch = data.get("products") or []
        if not batch:
            break
        out += batch
    seen, res = set(), []
    for p in out:
        if p["handle"] not in seen:
            seen.add(p["handle"])
            res.append(p)
    return res


def cell_text(c):
    c = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h\d>", "\n", c)
    c = H.unescape(re.sub(r"<[^>]+>", "", c)).replace("\xa0", " ")
    return [re.sub(r"\s+", " ", x).strip() for x in c.split("\n") if x.strip()]


def table_rows(body):
    out = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", body or "", re.S | re.I):
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S | re.I)
        if len(tds) >= 2:
            k = " ".join(cell_text(tds[0])).strip().rstrip(":").strip().lower()
            if k and k not in out:
                out[k] = cell_text(tds[1])
    return out


def cap(s):
    return s[:1].upper() + s[1:] if s else s


def get_material(r):
    for k, v in r.items():
        if "chất liệu" in k:
            t = " ".join(v)
            t = re.sub(r"^.*?chất liệu\s*:\s*", "", t, flags=re.I)
            t = t.lstrip("+-• ").strip()
            return cap(re.split(r"[,.;]", t)[0].strip())
    return ""


def get_size(r):
    for k, v in r.items():
        if "kích cỡ" in k or "kích thước" in k or "số đo" in k:
            t = ", ".join(x.strip(" ,.") for x in v if x.strip(" ,."))
            t = re.sub(r"(\dcm)\s*(?=[A-ZÀ-Ỹ])", r"\1, ", t)
            return t
    return ""


def get_form(r):
    for k, v in r.items():
        if "miêu tả" in k or "mô tả" in k:
            t = re.split(r"(?<=[.!])\s|\.(?=[A-ZÀ-Ỹ])", " ".join(v))[0].strip(" .")
            t = re.sub(r"^(Chân váy|Áo dài|Áo|Đầm|Váy|Quần|Blazer|Set)( kiểu)?\s+", "", t)
            return t[:1].lower() + t[1:] if t else ""
    return ""


def guess_cat(p):
    if p["product_type"] in TYPE_CAT:
        return TYPE_CAT[p["product_type"]]
    n = p["title"].lower()
    for cat, pat in [("acc", r"^(cài|kẹp|nịt|thắt lưng|túi|khăn|bông tai|vòng|dây chuyền|mũ|nón|băng đô)"),
                     ("other", r"^(áo dài|blazer|vest|đầm|set|bộ |jumpsuit|áo khoác|yếm)"),
                     ("bot", r"^(chân váy|váy|quần|jean|maxi|short)"),
                     ("top", r"^(áo|sơ mi|croptop|babytee|top)")]:
        if re.match(pat, n):
            return cat
    return "other"


def big(url):
    # ảnh cỡ 480px của Haravan: đủ nét, tải nhanh
    return re.sub(r"(\.\w+)(\?.*)?$", r"_large\1", url)


# ---------- mô hình nhìn hình ----------
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
_sess = None


def session():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        path = os.path.join(MODEL_DIR, "model_quantized.onnx")
        if not os.path.exists(path):
            os.makedirs(MODEL_DIR, exist_ok=True)
            log("Tải mô hình nhìn hình...")
            with open(path + ".tmp", "wb") as f:
                f.write(fetch(MODEL_URL))
            os.replace(path + ".tmp", path)
        _sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _sess


def embed(im):
    # giống AutoProcessor của dinov2: cạnh ngắn 256, cắt giữa 224, chuẩn hoá ImageNet
    w, h = im.size
    r = 256 / min(w, h)
    im = im.resize((max(1, round(w * r)), max(1, round(h * r))), Image.BICUBIC)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    a = (np.asarray(im.crop((l, t, l + 224, t + 224)), np.float32) / 255 - MEAN) / STD
    v = session().run(None, {"pixel_values": a.transpose(2, 0, 1)[None]})[0][0, 0]
    v = v / (np.abs(v).max() or 1) * 127
    return base64.b64encode(np.round(v).astype(np.int8).tobytes()).decode()


def crops(im, kind):
    W, Hh = im.size
    if kind == 1:   # phần áo, giống crop trên trang
        return im.crop((0, 0, W, min(Hh, round(Hh * 0.55) + 1)))
    if kind == 2:   # phần váy / quần
        return im.crop((0, round(Hh * 0.45), W, Hh))
    return im


def thumb(im):
    t = ImageOps.fit(im, THUMB, Image.LANCZOS, centering=(0.5, 0.4))
    b = io.BytesIO()
    t.save(b, "WEBP", quality=55, method=6)
    return "data:image/webp;base64," + base64.b64encode(b.getvalue()).decode()


# ---------- trang ----------
def read_const(lines, name):
    pre = f"const {name}="
    for i, ln in enumerate(lines):
        if ln.startswith(pre):
            return i, json.loads(ln[len(pre):].rstrip("\r").rstrip(";"))
    raise SystemExit(f"Không thấy {pre} trong index.html")


def write_const(lines, i, name, obj):
    cr = "\r" if lines[i].endswith("\r") else ""
    lines[i] = f"const {name}=" + json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + ";" + cr


def main():
    text = open(PAGE, encoding="utf-8", newline="").read()
    lines = text.split("\n")
    iD, oldD = read_const(lines, "D")
    iF, FORM = read_const(lines, "FORM")
    iI, oldIDX = read_const(lines, "IDX")
    old_h = {x["h"] for x in oldD}

    log("Đọc sản phẩm từ realclothes.co...")
    P = all_products()
    log(f"  {len(P)} sản phẩm trên web (trang đang có {len(oldD)})")
    P = [p for p in P if p.get("variants")]
    if len(P) < max(20, len(oldD) * 0.6):
        raise SystemExit("Web trả về quá ít sản phẩm, có thể đang lỗi. Giữ nguyên trang cũ.")
    P.sort(key=lambda p: p["handle"])

    cache = {}
    if os.path.exists(CACHE_FILE):
        try:
            cache = json.load(open(CACHE_FILE, encoding="utf-8"))
        except Exception:
            cache = {}
    used = set()

    # tải ảnh song song (chỉ ảnh chưa có trong cache)
    need = {}
    for p in P:
        cat = guess_cat(p)
        kinds = [0] + ([1] if cat == "top" or p["product_type"] == "Blazer" else []) + ([2] if cat == "bot" else [])
        for j, img in enumerate(p["images"][:MAX_IMG]):
            key = img["src"].split("?")[0]
            c = cache.get(key, {})
            if (j == 0 and "i" not in c) or any(str(k) not in c for k in kinds):
                need[key] = True
    log(f"  {len(need)} ảnh mới cần xử lý, {len(cache)} ảnh có sẵn trong bộ nhớ đệm")

    def load(key):
        # giữ dạng nén cho nhẹ bộ nhớ, lúc dùng mới mở ra
        for url in (big(key), key):
            try:
                b = fetch(url)
                Image.open(io.BytesIO(b)).verify()
                return key, b
            except Exception as e:
                err = e
        log(f"  bỏ qua ảnh lỗi {key}: {err}")
        return key, None

    images = {}
    with ThreadPoolExecutor(8) as ex:
        for n, (k, b) in enumerate(ex.map(load, list(need)), 1):
            if b is not None:
                images[k] = b
            if n % 500 == 0:
                log(f"  đã tải {n}/{len(need)} ảnh")

    D, IDX_p, IDX_c, IDX_d = [], [], [], []
    done = 0
    for p in P:
        r = table_rows(p.get("body_html"))
        names = [o["name"] for o in p.get("options") or []]
        ci = names.index("Màu sắc") if "Màu sắc" in names else -1
        si = names.index("Kích thước") if "Kích thước" in names else -1
        opt = lambda v, i: "" if i < 0 else (v.get(f"option{i + 1}") or "")
        V = [[opt(v, ci), opt(v, si), int(round(float(v["price"]))), 1 if v.get("available", True) else 0]
             for v in p["variants"]]
        prices = [v[2] for v in V]
        comp = [int(round(float(v.get("compare_at_price") or 0))) for v in p["variants"]]
        cat = guess_cat(p)
        kinds = [0] + ([1] if cat == "top" or p["product_type"] == "Blazer" else []) + ([2] if cat == "bot" else [])
        pid = len(D)
        thumb_uri = ""
        for j, img in enumerate(p["images"][:MAX_IMG]):
            key = img["src"].split("?")[0]
            c = cache.setdefault(key, {})
            raw = images.pop(key, None)
            im = Image.open(io.BytesIO(raw)).convert("RGB") if raw else None
            if j == 0 and "i" not in c and im is not None:
                c["i"] = thumb(im)
            for k in kinds:
                if str(k) not in c and im is not None:
                    c[str(k)] = embed(crops(im, k))
                    done += 1
                    if done % 1000 == 0:
                        log(f"  đã tính {done} vector hình")
                if str(k) in c:
                    IDX_p.append(pid); IDX_c.append(k); IDX_d.append(c[str(k)])
            if j == 0:
                thumb_uri = c.get("i", "")
            used.add(key)
        D.append({"h": p["handle"], "n": p["title"].strip(), "t": p["product_type"] or "Khác", "cat": cat,
                  "sale": "SALE" in p["title"].upper(),
                  "o": f"{max(comp) // 1000}k" if max(comp) > max(prices) else "",
                  "m": get_material(r), "pmin": min(prices), "pmax": max(prices), "v": V,
                  "i": thumb_uri, "z": get_size(r)})
        if p["handle"] not in FORM and p["handle"] not in old_h and cat != "acc":
            f = get_form(r)
            if f:
                FORM[p["handle"]] = f
                log(f"  món mới: {p['title']} (form: {f})")
    log(f"  đã tính {done} vector hình mới")

    # bỏ ảnh không còn dùng khỏi bộ nhớ đệm
    cache = {k: v for k, v in cache.items() if k in used}
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    json.dump(cache, open(CACHE_FILE, "w", encoding="utf-8"))

    IDX = {k: v for k, v in oldIDX.items() if k not in ("p", "c", "d")}
    IDX.update({"p": IDX_p, "c": IDX_c, "d": base64.b64encode(b"".join(base64.b64decode(x) for x in IDX_d)).decode()})

    if D == oldD and IDX == oldIDX:
        log("Dữ liệu không đổi so với trang hiện tại.")
        return

    write_const(lines, iD, "D", D)
    write_const(lines, iF, "FORM", FORM)
    write_const(lines, iI, "IDX", IDX)
    out = "\n".join(lines)
    today = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).strftime("%d/%m/%Y")
    out = re.sub(r"\d+ sản phẩm, dữ liệu ngày \d{1,2}/\d{1,2}/\d{4}", f"{len(D)} sản phẩm, dữ liệu ngày {today}", out, count=1)
    new_h = {x["h"] for x in D}
    log(f"Xong: {len(D)} sản phẩm. Thêm {len(new_h - old_h)}, bỏ {len(old_h - new_h)}.")
    for h in sorted(new_h - old_h):
        log("  + " + h)
    for h in sorted(old_h - new_h):
        log("  - " + h)
    with open(PAGE, "w", encoding="utf-8", newline="") as f:
        f.write(out)


if __name__ == "__main__":
    main()
