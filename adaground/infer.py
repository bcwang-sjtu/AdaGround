"""AdaGround inference: the six model stages of training-free GUI grounding with Qwen3-VL (vLLM).

Each invocation runs one stage (--mode) on one benchmark and writes one JSON result file: a summary and one record per
sample. The stages are combined offline (CSG header rule, existence-gated abstention) and scored there.

Stages
  zoom     fixed 4x crop chain: point on the whole screen (budget --b1), then point on a --crop_wh window centred on it,
           up-sampled to --b2 times its pixel count. The first pass alone is the single-pass baseline.
  exist    existence query: probability that the instruction's target is present on the screen (Yes/No).
  program  CSG: the model rewrites the instruction into a small program; a spreadsheet cell is located from its column
           and row headers. Every other sample keeps the answer of --retry_from (the zoom result).
  tsn      TNZ chain: coarse point and target box; while the target spans fewer than --tsn_tok_target visual tokens, zoom
           on a crop around the point (at most --tsn_depth zooms); a zoom is accepted only if its verification score
           does not drop.
  choose2  arbitration: the final points of the --choose_from files compete; when they differ by more than --mv_tau
           px, the candidate with the highest verification score wins (ties: file order).
  retry    negative-feedback re-grounding: the final point of --retry_from is verified; the sample is re-grounded with
           the rejected point marked by a red X and a note; the new answer replaces the old one when the old score is
           below tau and the new score is higher. One output file per tau in --retry_taus.

Verification score: probability of "Yes" (first-token logprobs) for a close-up of the candidate with a red circle on it.

Image views follow the standard Qwen3-VL preprocessing: a view with budget b > 1 is first up-sampled by sqrt(b); it is
then resized with qwen-vl-utils' smart_resize on the processor grid (patch size x merge size) under --max_pixels, which
is the resolution stated in the locator prompt, converted to RGB and resized by qwen-vl-utils on its 28-pixel grid;
vLLM's processor performs the final resize. Locator answers are parsed as the first "(x, y)" integer pair on the 0-1000
grid of the view; an unparsable answer gives no point and every stage keeps its own fallback (zoom keeps pass 1,
re-grounding pass 3 -> pass 2 -> pass 1, CSG stage B -> stage A, TNZ no zoom).
"""
import argparse, collections, json, logging, math, os, re, time

# the HF processor is loaded before the vLLM engine starts; a forked engine process would deadlock
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from PIL import Image  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

Image.MAX_IMAGE_PIXELS = None
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("adaground")

# ----------------------------------------------------------------------------- prompts
QWEN3_SYSTEM = (
    "You are an expert UI element locator. Given a GUI image and a user's "
    "element description, provide the coordinates of the specified element "
    "as a single (x,y) point. The image resolution is height {height} and "
    "width {width}. For elements with area, return the center point.\n\n"
    "Output the coordinate pair exactly:\n(x,y)"
)

# ----------------------------------------------------------------------------- image views
CAP_PIXELS = 8847360                # pixel cap of every view and of vLLM's processor (--max_pixels)
MIN_PIXELS = 3136
QVU_FACTOR = 28                     # qwen-vl-utils grid: patch 14 x spatial merge 2
QVU_MIN_PIXELS = 4 * 28 * 28        # 3136
QVU_MAX_PIXELS = 16384 * 28 * 28    # 12845056
PROC_MIN_PIXELS = 3136              # pixel range the HF processor is loaded with
PROC_MAX_PIXELS = 4096 * 2160       # 8847360


def open_image(path):
    """Decode a screenshot in the file's own mode; render() converts it to RGB after the first resize."""
    im = Image.open(path)
    im.load()
    return im


def qvu_smart_resize(height, width, factor, min_pixels, max_pixels):
    """qwen-vl-utils 0.0.14 smart_resize (round to the grid with Python's round, floor under the cap, ceil over the
    floor); returns (height, width)."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def qvu_to_rgb(im):
    """qwen-vl-utils to_rgb: RGBA composited on white, every other mode convert('RGB')."""
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        return bg
    return im.convert("RGB")


def render(img, budget, runner):
    """View of `img` (screenshot or crop) at `budget` handed to vLLM, and the (w, h) stated in the locator prompt.
    budget > 1: up-sample by sqrt(budget) with PIL's default filter; budget < 1: the first resize is capped at budget x
    the pixel count of the full-resolution view. First resize: smart_resize on the processor grid (stated size); second
    resize: qwen-vl-utils' 28-grid smart_resize."""
    f = runner.proc_factor
    W, H = img.size
    cap = CAP_PIXELS
    if budget > 1.0:
        sc = math.sqrt(budget)
        z = img.resize((int(round(W * sc)), int(round(H * sc))))
    else:
        z = img
        if budget < 1.0:
            fh, fw = qvu_smart_resize(H, W, f, MIN_PIXELS, CAP_PIXELS)
            cap = max(MIN_PIXELS, int(fw * fh * budget))
    h1, w1 = qvu_smart_resize(z.height, z.width, f, MIN_PIXELS, cap)
    v1 = qvu_to_rgb(z.resize((w1, h1)))
    h2, w2 = qvu_smart_resize(h1, w1, QVU_FACTOR, QVU_MIN_PIXELS, QVU_MAX_PIXELS)
    return v1.resize((w2, h2)), (w1, h1)


def view_pt(pt, W, H):
    """A point drawn on a view (verification circle, exclusion mark) is clamped to [0, W-1] x [0, H-1]; the answers
    themselves stay unclamped. Crop windows need no clamping: crop_box/crop_rect clamp the placement."""
    if pt is None:
        return pt
    return (min(max(pt[0], 0), W - 1), min(max(pt[1], 0), H - 1))


def audit_rec(r):
    """Raw answer, parse status and stated size of one locator pass (per-sample records of the composite stages)."""
    return {k: r[k] for k in ("raw", "parse", "stated") if k in r} if r else None


# ----------------------------------------------------------------------------- geometry
def adaptive_d(W, H, eta, dmin, dmax):
    return int(min(max(math.floor(math.sqrt(W * H / eta)), dmin), dmax))


def crop_box(W, H, cx, cy, d):
    wc, hc = min(d, W), min(d, H)
    x0 = int(min(max(cx - wc / 2, 0), W - wc))
    y0 = int(min(max(cy - hc / 2, 0), H - hc))
    return [x0, y0, wc, hc]


def crop_rect(W, H, cx, cy, cw, ch):
    """[x0, y0, w, h] of a cw x ch window centred on (cx, cy), clipped to and shifted inside the screen."""
    wc, hc = int(min(cw, W)), int(min(ch, H))
    x0 = int(min(max(cx - wc / 2, 0), W - wc))
    y0 = int(min(max(cy - hc / 2, 0), H - hc))
    return [x0, y0, wc, hc]


def p2_box(a, W, H, cx, cy):
    """Zoom pass-2 window: the fixed --crop_wh rectangle centred on the pass-1 point."""
    cw, ch = [int(v) for v in a.crop_wh.split(",")]
    return crop_rect(W, H, cx, cy, cw, ch)


def contains(box, bbox):
    x0, y0, wc, hc = box
    return x0 <= bbox[0] and y0 <= bbox[1] and bbox[2] <= x0 + wc and bbox[3] <= y0 + hc


def in_bbox(p, bbox):
    return p is not None and bbox[0] <= p[0] <= bbox[2] and bbox[1] <= p[1] <= bbox[3]


# ----------------------------------------------------------------------------- parsing
POINT_RE = re.compile(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)")


def hf_clean_up_tokenization(s):
    """transformers PreTrainedTokenizerBase.clean_up_tokenization (decoding with clean_up_tokenization_spaces=True)."""
    return (s.replace(" .", ".").replace(" ?", "?").replace(" !", "!").replace(" ,", ",").replace(" ' ", "'")
            .replace(" n't", "n't").replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve").replace(" 're", "'re"))


def extract_point(raw):
    """Every "(a,b)" of the cleaned-up answer converted with int(), the first one returned; None when there is no match
    or any match is not an integer."""
    try:
        return [tuple(map(int, m)) for m in POINT_RE.findall(hf_clean_up_tokenization(raw))][0]
    except Exception:
        return None


# ----------------------------------------------------------------------------- datasets
def load_sspro(data_dir):
    out = []
    ann = os.path.join(data_dir, "annotations")
    for jf in sorted(os.listdir(ann)):
        if not jf.endswith(".json"):
            continue
        for it in json.load(open(os.path.join(ann, jf))):
            out.append(dict(id=f'{jf[:-5]}/{it["img_filename"]}/{len(out)}', task=jf[:-5],
                            app=it.get("application", jf[:-5]), ui_type=it.get("ui_type", "?"),
                            group=it.get("group", "?"),
                            img_path=os.path.join(data_dir, "images", it["img_filename"]),
                            screenshot=it["img_filename"], instruction=it["instruction"], bbox=it["bbox"]))
    return out


def load_osworldg(data_dir):
    """OSWorld-G with the refined instructions. Refusal samples (no target) get the all-zero box."""
    jf = os.path.join(data_dir, "OSWorld-G_refined.json")
    out = []
    for it in json.load(open(jf)):
        bc, bt = it["box_coordinates"], it.get("box_type", "bbox")
        if bt == "polygon" and len(bc) >= 6:
            xs, ys = bc[0::2], bc[1::2]
            bbox = [min(xs), min(ys), max(xs), max(ys)]
        elif bt == "refusal" or len(bc) < 4:
            bbox = [0.0, 0.0, 0.0, 0.0]
        else:
            x, y, w, h = bc[:4]
            bbox = [x, y, x + w, y + h]
        out.append(dict(id=it["id"], task=",".join(it.get("GUI_types", ["unk"])[:1]), app="osworld",
                        ui_type=bt, group="osworld", img_path=os.path.join(data_dir, "images", it["image_path"]),
                        screenshot=it["image_path"], instruction=it["instruction"], bbox=bbox))
    return out


def load_ssv2(data_dir):
    img_dir = os.path.join(data_dir, "images", "screenspotv2_image")
    if not os.path.isdir(img_dir):
        img_dir = os.path.join(data_dir, "screenspotv2_image")
    out = []
    for split in ["desktop", "mobile", "web"]:
        for it in json.load(open(os.path.join(data_dir, f"screenspot_{split}_v2.json"))):
            x, y, w, h = it["bbox"]
            out.append(dict(id=f'{split}/{it["img_filename"]}/{len(out)}', task=split, app=split,
                            ui_type=it.get("data_type", "?"), group=split,
                            img_path=os.path.join(img_dir, it["img_filename"]), screenshot=it["img_filename"],
                            instruction=it["instruction"], bbox=[x, y, x + w, y + h]))
    return out


def load_uivision(data_dir, splits=("basic", "functional", "spatial")):
    out = []
    for sp in splits:
        jf = os.path.join(data_dir, "annotations", "element_grounding", f"element_grounding_{sp}.json")
        for it in json.load(open(jf)):
            out.append(dict(id=f'{sp}/{it["image_path"]}/{len(out)}', task=sp, app=it.get("platform", "?"),
                            ui_type=it.get("element_type", "?"), group=it.get("category", "?"),
                            img_path=os.path.join(data_dir, "images", it["image_path"]),
                            screenshot=it["image_path"], instruction=it["prompt_to_evaluate"], bbox=it["bbox"]))
    return out


LOADERS = {"sspro": load_sspro, "osworldg": load_osworldg, "ssv2": load_ssv2, "uivision": load_uivision}


# ----------------------------------------------------------------------------- runner
class Runner:
    """vLLM engine of one Qwen3-VL backbone, prompt construction, generation and point decoding."""

    def __init__(self, a):
        self.a = a
        self.last_parse = None     # parse status of the answer decoded last ("ok" | "fail")
        self._check_processor()
        kw = {}
        if a.max_num_seqs > 0:
            kw["max_num_seqs"] = a.max_num_seqs
        self.llm = LLM(model=a.model_path, trust_remote_code=True, tensor_parallel_size=a.tp,
                       limit_mm_per_prompt={"image": 1}, max_model_len=a.max_model_len,
                       gpu_memory_utilization=a.gpu_mem_util, enforce_eager=True,
                       mm_processor_kwargs={"min_pixels": MIN_PIXELS, "max_pixels": CAP_PIXELS}, **kw)
        self.tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
        self.image_pad_id = self.tok.convert_tokens_to_ids("<|image_pad|>")
        stop = ["<|endoftext|>", "<|im_end|>"]      # generation stops at the end-of-turn tokens only
        self.greedy = SamplingParams(temperature=0.0, max_tokens=32, stop=stop)

    def _check_processor(self):
        """Loads the HF processor with the pixel range [PROC_MIN_PIXELS, PROC_MAX_PIXELS] and takes from it the grid the
        views are resized to (patch_size x merge_size, 32 for Qwen3-VL). Checks that the processor's effective pixel caps
        equal the ones the views are rendered with (MIN_PIXELS and --max_pixels, which vLLM receives as
        mm_processor_kwargs): its min_pixels/max_pixels attributes and the grids it gives two probe images (3200x3000
        binds the max cap, 100x100 the min cap). Then checks that the hand-built locator prompt equals the processor's
        chat template for [system, user[image, text]] with add_generation_prompt=True."""
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(self.a.model_path, trust_remote_code=True, min_pixels=PROC_MIN_PIXELS,
                                             max_pixels=PROC_MAX_PIXELS)
        ip = proc.image_processor
        self.proc_factor = int(ip.patch_size) * int(ip.merge_size)
        if self.a.factor and self.a.factor != self.proc_factor:
            raise ValueError(f"views are resized on the processor grid {self.proc_factor}; --factor {self.a.factor} differs")
        attr = (getattr(ip, "min_pixels", None), getattr(ip, "max_pixels", None))
        probes = {}
        for W, H in ((3200, 3000), (100, 100)):
            g = ip(images=[Image.new("RGB", (W, H), (90, 120, 200))], return_tensors="pt")["image_grid_thw"][0]
            h, w = qvu_smart_resize(H, W, self.proc_factor, MIN_PIXELS, CAP_PIXELS)
            probes[f"{W}x{H}"] = dict(grid=[int(v) for v in (g.tolist() if hasattr(g, "tolist") else g)],
                                      want=[1, h // int(ip.patch_size), w // int(ip.patch_size)])
        effective_ok = attr == (MIN_PIXELS, CAP_PIXELS) and all(p["grid"] == p["want"] for p in probes.values())
        if not effective_ok:
            raise ValueError(f"the processor (min_pixels={PROC_MIN_PIXELS}, max_pixels={PROC_MAX_PIXELS}) has effective "
                             f"caps {attr}, probe grids {probes}; the views are rendered with [{MIN_PIXELS}, "
                             f"{CAP_PIXELS}] (--max_pixels): run with --max_pixels {PROC_MAX_PIXELS}")
        probe = "Click the probe button"
        msgs = [{"role": "system", "content": QWEN3_SYSTEM.format(height=1472, width=2560)},
                {"role": "user", "content": [{"type": "image", "image": "probe.png"}, {"type": "text", "text": probe}]}]
        ref = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ours = self.build(Image.new("RGB", (64, 64)), probe, stated=(2560, 1472))["prompt"]
        if ours != ref:
            raise ValueError(f"the hand-built locator prompt differs from the chat template of "
                             f"{self.a.model_path}\nours:     {ours!r}\ntemplate: {ref!r}")
        logger.info(f"locator prompt == processor.apply_chat_template; processor grid {self.proc_factor}, "
                    f"processor pixels [{MIN_PIXELS}, {CAP_PIXELS}]")

    # --- prompt construction -------------------------------------------------
    def build(self, screenshot_view, instruction, stated=None):
        """Locator query: system prompt stating the resolution `stated` (w, h) (default: the view's size), the image,
        then the instruction. Returns the vLLM input dict."""
        vw, vh = stated or screenshot_view.size
        shot = "<|vision_start|><|image_pad|><|vision_end|>"
        sys_text = QWEN3_SYSTEM.format(height=vh, width=vw)
        prompt = (f"<|im_start|>system\n{sys_text}<|im_end|>\n<|im_start|>user\n{shot}"
                  f"{instruction}<|im_end|>\n<|im_start|>assistant\n")
        return {"prompt": prompt, "multi_modal_data": {"image": screenshot_view}}

    VERIFY_Q = ("The red circle marks a candidate click position in this region of a screenshot. Instruction: \"{instr}\". "
                "Is the red circle on the UI element that this instruction refers to? Answer Yes or No.")

    def build_verify(self, view, instruction):
        q = self.VERIFY_Q.format(instr=instruction)
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                  "<|vision_start|><|image_pad|><|vision_end|>" + q + "<|im_end|>\n<|im_start|>assistant\n")
        return {"prompt": prompt, "multi_modal_data": {"image": view}}

    def verify(self, inputs):
        """Returns list of dict(p_yes, mass, lp_yes, lp_no, img_tokens) from the first-token Yes/No logprobs."""
        from vllm import SamplingParams
        if not hasattr(self, "_yn"):
            def ids(words):
                out = set()
                for w in words:
                    enc = self.tok.encode(w, add_special_tokens=False)
                    if len(enc) == 1:
                        out.add(enc[0])
                return out
            self._yn = (ids(["Yes", "yes", "YES"]), ids(["No", "no", "NO"]))
            logger.info(f"verify: yes ids {sorted(self._yn[0])} no ids {sorted(self._yn[1])}")
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        res = self.llm.generate(inputs, sampling_params=sp, use_tqdm=False)
        out = []
        for r in res:
            lp = r.outputs[0].logprobs[0] if r.outputs[0].logprobs else {}
            py = sum(math.exp(v.logprob) for k, v in lp.items() if k in self._yn[0])
            pn = sum(math.exp(v.logprob) for k, v in lp.items() if k in self._yn[1])
            floor = min((v.logprob for v in lp.values()), default=-50.0)   # absent from top-k: bounded by the k-th logprob
            ids = r.prompt_token_ids or []
            out.append(dict(p_yes=py / (py + pn) if py + pn > 0 else 0.5, mass=py + pn,
                            lp_yes=math.log(py) if py > 0 else floor, lp_no=math.log(pn) if pn > 0 else floor,
                            img_tokens=sum(1 for t in ids if t == self.image_pad_id)))
        return out

    def generate(self, inputs):
        """Greedy generation. Returns list of dict(text, img_tokens, prompt_tokens, gen_tokens)."""
        res = self.llm.generate(inputs, sampling_params=self.greedy, use_tqdm=False)
        out = []
        for r in res:
            ids = r.prompt_token_ids or []
            out.append(dict(text=r.outputs[0].text.strip(),
                            img_tokens=sum(1 for t in ids if t == self.image_pad_id),
                            prompt_tokens=len(ids), gen_tokens=sum(len(o.token_ids) for o in r.outputs)))
        return out

    def decode(self, text, frame):
        """Locator answer -> point in original pixels. (a, b) = extract_point(text) on the 0-1000 grid of frame =
        (x0, y0, w, h), the rendered screen or crop in original pixels; point = (int(a / 1000 * w + x0),
        int(b / 1000 * h + y0)), not clamped. None when unparsable. last_parse keeps the status (ok | fail)."""
        xy, st = extract_point(text), "ok"
        if xy is None:
            st = "fail"
        self.last_parse = st
        if xy is None:
            return None
        x0, y0, fw, fh = frame
        return (int(xy[0] / 1000 * fw + x0), int(xy[1] / 1000 * fh + y0))

    def audit_fields(self, m):
        """Per-call record fields of a locator pass: parse status of the answer just decoded and the resolution stated
        in the prompt."""
        out = {"parse": self.last_parse}
        if m.get("stated"):
            out["stated"] = [int(v) for v in m["stated"]]
        return out


# ----------------------------------------------------------------------------- job specs (lazy, streaming)
class VerifySpec:
    """Crop of side `side` (original px) centred on candidate point, up-sampled by budget b, with a red circle at the point."""

    def __init__(self, sample, point, side, b):
        self.s, self.pt, self.side, self.b = sample, point, side, b

    def materialize(self, runner):
        from PIL import ImageDraw
        s = self.s
        img = open_image(s["img_path"])
        W, H = img.size
        pt = view_pt(self.pt, W, H)
        x0, y0, wc, hc = crop_box(W, H, pt[0], pt[1], self.side)
        crop = img.crop((x0, y0, x0 + wc, y0 + hc))
        view = render(crop, self.b, runner)[0]
        sx, sy = view.size[0] / wc, view.size[1] / hc
        cx, cy = (pt[0] - x0) * sx, (pt[1] - y0) * sy
        r = max(10, int(0.025 * max(view.size)))
        dr = ImageDraw.Draw(view)
        dr.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(255, 0, 0), width=max(2, r // 4))
        return runner.build_verify(view, s["instruction"]), dict(box=[x0, y0, wc, hc], view_size=list(view.size))


def run_verify(runner, specs, chunk):
    """Yes/No verification of every spec (VerifySpec or ExistSpec), streamed by chunk; returns key -> verify() result
    plus the spec's view metadata."""
    keys = list(specs)
    res = {}
    for i in range(0, len(keys), chunk):
        batch = keys[i:i + chunk]
        mats = materialize_all(runner, specs, batch)
        inputs, metas = [x[0] for x in mats], [x[1] for x in mats]
        for k, m, o in zip(batch, metas, runner.verify(inputs)):
            res[k] = dict(o, **m)
        del inputs, metas
    return res


def group_points(points, tau):
    """Complete-linkage grouping: a point joins a group only if within tau px of ALL its members.
    points: list of (name, (x, y)). Returns groups sorted by size desc (stable w.r.t. input order)."""
    groups = []
    for name, p in points:
        if p is None:
            continue
        for g in groups:
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) <= tau for _, q in g):
                g.append((name, p))
                break
        else:
            groups.append([(name, p)])
    return sorted(groups, key=len, reverse=True)


class JobSpec:
    """One locator call without holding pixels; materialize() decodes the image on demand.
    kind: "shot" (whole screen at budget b) | "crop" (box [x0, y0, w, h] of the original at budget b).
    exclude: a rejected point marked with a red X on the whole-screen view (re-grounding). inst: prompt instruction
    override (scoring always uses the sample)."""

    def __init__(self, sample, kind, b, box=None, exclude=None, inst=None):
        self.s, self.kind, self.b, self.box, self.exclude = sample, kind, b, box, exclude
        self.inst = inst

    def materialize(self, runner):
        s = self.s
        img = open_image(s["img_path"])
        W, H = img.size
        if self.kind == "shot":
            view, stated = render(img, self.b, runner)
            vw, vh = view.size
            frame = (0, 0, W, H)
            if self.exclude:
                from PIL import ImageDraw
                xp, yp = view_pt(self.exclude, W, H)
                ex, ey = xp * vw / W, yp * vh / H
                r = max(8, int(0.012 * max(vw, vh)))
                dr = ImageDraw.Draw(view)
                for dx in (-1, 1):
                    dr.line([ex - r, ey - dx * r, ex + r, ey + dx * r], fill=(255, 0, 0), width=max(3, r // 3))
        else:
            x0, y0, wc, hc = self.box
            crop = img.crop((x0, y0, x0 + wc, y0 + hc))
            view, stated = render(crop, self.b, runner)
            vw, vh = view.size
            frame = (x0, y0, wc, hc)
        inp = runner.build(view, self.inst or s["instruction"], stated=stated)
        return inp, dict(view_size=[vw, vh], frame=frame, stated=stated)


_MAT_POOL = None


def materialize_all(runner, specs, keys):
    """Decode/crop/resize images for a batch in parallel (PIL releases the GIL); returns list of (input, meta)."""
    global _MAT_POOL
    if _MAT_POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _MAT_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("MAT_THREADS", "12")))
    return list(_MAT_POOL.map(lambda k: specs[k].materialize(runner), keys))


def run_pass(runner, specs, chunk):
    """specs: dict key -> JobSpec. Streams by chunk; returns key -> result dict (no pixels kept), including the parse
    status and stated size of the answer (Runner.audit_fields)."""
    keys = list(specs)
    res = {}
    for i in range(0, len(keys), chunk):
        batch = keys[i:i + chunk]
        mats = materialize_all(runner, specs, batch)
        inputs, metas = [x[0] for x in mats], [x[1] for x in mats]
        outs = runner.generate(inputs)
        for k, m, o in zip(batch, metas, outs):
            pt = runner.decode(o["text"], m["frame"])
            res[k] = dict(point=pt, raw=o["text"][:160], img_tokens=o["img_tokens"], prompt_tokens=o["prompt_tokens"],
                          gen_tokens=o["gen_tokens"], view_size=m["view_size"], ref_meta=None)
            res[k].update(runner.audit_fields(m))
        del inputs, metas
        done = min(i + chunk, len(keys))
        if done % (chunk * 10) < chunk or done == len(keys):
            logger.info(f"  generated {done}/{len(keys)}")
    return res


# ----------------------------------------------------------------------------- arbitration (choose2)
def run_choose2(a, runner, samples, meta, t0, timings):
    """Candidates = FINAL points of the result files in --choose_from (comma list; file order = priority). Points within
    --mv_tau px are grouped and each group is represented by its first member. With two or more groups, the
    representatives of the --mv_k largest groups are verified (close-up of side --mv_vside x d at budget --b2) and the
    one with the highest P(Yes) wins (ties: file order); otherwise the first file's point is kept.
    Visual tokens: those of all input files plus the verification views."""
    files = [f for f in a.choose_from.split(",") if f]
    R = [{d["id"]: d for d in json.load(open(f))["details"]} for f in files]
    cspecs, cands = {}, {}
    for s in samples:
        sid = s["id"]
        pts = [(k, tuple(r[sid]["final"])) for k, r in enumerate(R) if r[sid].get("final")]
        groups = group_points(pts, a.mv_tau)
        reps = [g[0] for g in groups]          # first member of each group (file order = priority)
        cands[sid] = reps
        if len(reps) >= 2:
            cspecs[sid] = (s, [p for _, p in reps[:a.mv_k]], meta[sid]["d"])
    ta = time.time()
    vspecs = {}
    for sid, (s, pts, d) in cspecs.items():
        for j, pt in enumerate(pts):
            vspecs[f"{sid}||{j}"] = VerifySpec(s, pt, int(a.mv_vside * d), a.b2)
    vr = run_verify(runner, vspecs, a.chunk)
    ver = {sid: [vr[f"{sid}||{j}"] for j in range(len(pts))] for sid, (s, pts, d) in cspecs.items()}
    timings = dict(timings, choose_s=time.time() - ta)
    p1, p2 = {}, {}
    meta_r = {k: dict(v) for k, v in meta.items()}
    for s in samples:
        sid = s["id"]
        base = R[-1][sid]
        tok = sum(r[sid]["vis_tokens"] for r in R)
        reps = cands[sid]
        v = ver.get(sid)
        if v:
            K = len(v)
            vp = [e["p_yes"] for e in v]
            k, pt = reps[max(range(K), key=lambda j: vp[j])]
            meta_r[sid]["choose"] = dict(probs=None, p_yes=[round(x, 4) for x in vp], picked_file=k)
            vt = sum(e["img_tokens"] for e in v)
            p1[sid] = dict(point=pt, raw="", img_tokens=tok + vt, prompt_tokens=0, gen_tokens=0, view_size=None,
                           ref_meta=None)
            meta_r[sid]["crop"] = base.get("crop")
            continue
        k, pt = reps[0] if reps else (None, None)
        p1[sid] = dict(point=pt, raw="", img_tokens=tok, prompt_tokens=0, gen_tokens=0, view_size=None, ref_meta=None)
        meta_r[sid]["crop"] = base.get("crop")
    finalize(a, "choose2", samples, meta_r, p1, p2, t0, timings, a.output,
             extra=dict(choose_from=files, mv_tau=a.mv_tau, choose_tok=a.choose_tok))


# ----------------------------------------------------------------------------- existence query (exist)
EXIST_Q = ("Instruction: \"{instr}\".\nIs there a UI element on this screen that this instruction refers to, so that the "
           "instruction can be carried out by clicking it? Answer Yes or No.")


class ExistSpec:
    """Existence check on the whole screen (budget b): does the instruction's target exist?"""

    def __init__(self, sample, b):
        self.s, self.b = sample, b

    def materialize(self, runner):
        s = self.s
        img = open_image(s["img_path"])
        view = render(img, self.b, runner)[0]
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                  "<|vision_start|><|image_pad|><|vision_end|>" + EXIST_Q.format(instr=s["instruction"]) +
                  "<|im_end|>\n<|im_start|>assistant\n")
        return {"prompt": prompt, "multi_modal_data": {"image": view}}, dict(view_size=list(view.size))


def run_exist(a, runner, samples):
    specs = {s["id"]: ExistSpec(s, a.b1) for s in samples}
    res = run_verify(runner, specs, a.chunk)
    out = [dict(id=s["id"], p_exist=round(res[s["id"]]["p_yes"], 5), yn_mass=round(res[s["id"]]["mass"], 4),
                img_tokens=res[s["id"]]["img_tokens"], refusal=s["bbox"] == [0.0, 0.0, 0.0, 0.0]) for s in samples]
    os.makedirs(os.path.dirname(os.path.abspath(a.output)), exist_ok=True)
    json.dump(dict(summary=dict(mode="exist", b1=a.b1, n=len(out)), details=out), open(a.output, "w"))
    logger.info(f"MODE=exist b1={a.b1} n={len(out)} mean p_exist {sum(x['p_exist'] for x in out) / len(out):.3f} "
                f"tokens {sum(x['img_tokens'] for x in out) / len(out):.0f}; saved {a.output}")


# ----------------------------------------------------------------------------- box questions
BOX_Q = ("Locate the UI element that this instruction refers to: \"{instr}\".\nOutput its bounding box as JSON "
         "[{{\"bbox_2d\": [x1, y1, x2, y2]}}] with coordinates on the 0-1000 grid of the image.")


class BoxQSpec:
    """Free-form box question on the whole screen (box=None, budget b) or on a crop box (x0, y0, w, h) at budget b.
    The answer's 0-1000 boxes are mapped back to original pixels."""

    def __init__(self, sample, question, b, box=None):
        self.s, self.q, self.b, self.box = sample, question, b, box

    def materialize(self, runner):
        img = open_image(self.s["img_path"])
        W, H = img.size
        if self.box:
            x0, y0, wc, hc = self.box
            view = render(img.crop((x0, y0, x0 + wc, y0 + hc)), self.b, runner)[0]
        else:
            x0, y0, wc, hc = 0, 0, W, H
            view = render(img, self.b, runner)[0]
        prompt = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
                  "<|vision_start|><|image_pad|><|vision_end|>" + self.q + "<|im_end|>\n<|im_start|>assistant\n")
        return {"prompt": prompt, "multi_modal_data": {"image": view}}, dict(frame=[x0, y0, wc, hc], view_size=list(view.size))


def parse_boxes(text, frame):
    """Every bbox_2d-like quadruple in the text, read on the 0-1000 grid of the view -> [x1, y1, x2, y2] in original
    pixels."""
    x0, y0, wc, hc = frame
    sx, sy = wc / 1000, hc / 1000
    out = []
    for m in re.finditer(r"\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\]", text):
        a1, b1, a2, b2 = [float(v) for v in m.groups()]
        a1, a2 = sorted((a1, a2)); b1, b2 = sorted((b1, b2))
        out.append([x0 + a1 * sx, y0 + b1 * sy, x0 + a2 * sx, y0 + b2 * sy])
    return out


def run_boxq(runner, specs, chunk, max_tokens):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, stop=["<|im_end|>", "<|endoftext|>"])
    keys, res = list(specs), {}
    for c in range(0, len(keys), chunk):
        batch = keys[c:c + chunk]
        inputs, metas = [], []
        for k in batch:
            inp, m = specs[k].materialize(runner)
            inputs.append(inp); metas.append(m)
        for k, m, o in zip(batch, metas, runner.llm.generate(inputs, sampling_params=sp, use_tqdm=False)):
            t = o.outputs[0].text
            res[k] = dict(raw=t[:400], boxes=parse_boxes(t, m["frame"]),
                          img_tokens=sum(1 for x in (o.prompt_token_ids or []) if x == runner.image_pad_id))
        del inputs, metas
    return res


# ----------------------------------------------------------------------------- CSG (program)
PROG_Q = """Convert a GUI click instruction into a JSON program for a grounding engine. Operations:
1. {"op": "ground"} - the instruction names or describes a UI element (button, icon, menu item, tab, field, link, list item, text, image, etc.) and clicking anywhere on that element is fine. Use this for almost all instructions.
2. {"op": "cell", "col": "<column letters>", "row": <row number>} - the instruction asks for one cell inside the data grid of a spreadsheet, given by its column and row, e.g. "cell C12" or "the 3rd column and 12th row" -> {"op": "cell", "col": "C", "row": 12}. Do NOT use it for a row header (row number button), a column header (column letter button), a cell corner or fill handle, a cell described only by its content, or a grid that is not a spreadsheet (color palette, shape gallery, icon table, calendar) - use ground for those.
3. {"op": "along", "element": "<short description of the long element>", "fx": <0..1>, "fy": <0..1>} - the instruction asks for a specific relative position on a slider, progress bar, scroll bar, divider line or another long thin element; fx/fy are the relative position inside that element's box, e.g. "the left end of the volume slider" -> fx 0.02, fy 0.5; "30% along the seek bar" -> fx 0.3, fy 0.5; "the bottom part of the vertical scroll bar" -> fx 0.5, fy 0.9; "the middle of the divider" -> fx 0.5, fy 0.5.
4. {"op": "text_edge", "anchor": "<exact visible text>", "side": "before" or "after", "context": "<where that text is>"} - the instruction asks to place the text cursor immediately before or after a specific visible piece of text, e.g. "put the cursor right after the word 'Total:' in the table header" -> {"op": "text_edge", "anchor": "Total:", "side": "after", "context": "table header"}.
Instruction: "<<INSTR>>"
Output only the JSON object."""


def parse_prog(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {"op": "ground"}
    try:
        p = json.loads(m.group(0))
    except Exception:
        return {"op": "ground", "bad": text[:120]}
    op = p.get("op")
    try:
        if op == "cell" and re.fullmatch(r"[A-Za-z]{1,3}", str(p.get("col", ""))) and int(p.get("row", 0)) > 0:
            return {"op": "cell", "col": str(p["col"]).upper(), "row": int(p["row"])}
        if op == "along" and p.get("element"):
            return {"op": "along", "element": str(p["element"]), "fx": min(1.0, max(0.0, float(p.get("fx", 0.5)))),
                    "fy": min(1.0, max(0.0, float(p.get("fy", 0.5))))}
        if op == "text_edge" and p.get("anchor") and p.get("side") in ("before", "after"):
            return {"op": "text_edge", "anchor": str(p["anchor"]), "side": p["side"], "context": str(p.get("context", ""))}
    except (TypeError, ValueError):
        pass
    return {"op": "ground"}


def run_program(a, runner, samples, meta, t0, timings):
    """Instruction-to-program grounding (CSG). The same model rewrites each instruction (text only) into one of:
    ground (keep the base FINAL of --retry_from) | cell(col, row): x from the column header, y from the row header (each:
    whole-screen point -> point on a tight view, side max(0.5 d, --retry_min_side) at budget --retry_b3) |
    along(element, fx, fy): element box (whole screen) -> refined box (view around it) -> relative position |
    text_edge(anchor, side): anchor-text box (whole screen) -> refined box (tight view) -> just before/after the text.
    Outputs: <stem>_all.json (every program op applied) and one file per op (<stem>_cell/_along/_text.json, only that op
    applied)."""
    from vllm import SamplingParams
    R = {d["id"]: d for d in json.load(open(a.retry_from))["details"]}
    S = {s["id"]: s for s in samples}
    keys = list(S)
    sp = SamplingParams(temperature=0.0, max_tokens=120, stop=["<|im_end|>", "<|endoftext|>"])
    ta = time.time()
    progs = {}
    for c in range(0, len(keys), 256):
        batch = keys[c:c + 256]
        inputs = [{"prompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n" +
                             PROG_Q.replace("<<INSTR>>", S[k]["instruction"].replace('"', "'")) + "<|im_end|>\n<|im_start|>assistant\n"} for k in batch]
        for k, o in zip(batch, runner.llm.generate(inputs, sampling_params=sp, use_tqdm=False)):
            progs[k] = parse_prog(o.outputs[0].text)
    timings = dict(timings, parse_s=time.time() - ta)
    ops = collections.Counter(p["op"] for p in progs.values())
    logger.info(f"programs: {dict(ops)}")
    side3 = lambda k: max(int(0.5 * meta[k]["d"]), a.retry_min_side)
    # stage A (whole screen)
    ptA, boxA = {}, {}
    for k, p in progs.items():
        s = S[k]
        if p["op"] == "cell":
            ptA[f"{k}||col"] = JobSpec(s, "shot", 1.0, inst=f'The column header labeled "{p["col"]}" at the top of the spreadsheet grid')
            ptA[f"{k}||row"] = JobSpec(s, "shot", 1.0, inst=f'The row header labeled "{p["row"]}" at the left of the spreadsheet grid')
        elif p["op"] == "along":
            q = BOX_Q.format(instr=f'{p["element"]} (instruction: {s["instruction"]})'.replace('"', "'"))
            boxA[k] = BoxQSpec(s, q, 1.0)
        elif p["op"] == "text_edge":
            q = (f"Locate the visible text '{p['anchor']}' ({p['context']}).\nOutput the bounding box of exactly that text as JSON "
                 "[{\"bbox_2d\": [x1, y1, x2, y2]}] with coordinates on the 0-1000 grid of the image.")
            boxA[k] = BoxQSpec(s, q, 1.0)
    rA = run_pass(runner, ptA, a.chunk) if ptA else {}
    bA = run_boxq(runner, boxA, a.chunk, 64) if boxA else {}
    # stage B (refinement views)
    ptB, boxB = {}, {}
    for key, spec in ptA.items():
        k = key.split("||")[0]
        pt = rA[key]["point"]
        if pt:
            W, H = S[k]["img_size"]
            ptB[key] = JobSpec(S[k], "crop", a.retry_b3, box=crop_box(W, H, pt[0], pt[1], side3(k)), inst=spec.inst)
    for k, spec in boxA.items():
        bx = bA[k]["boxes"][0] if bA[k]["boxes"] else None
        if not bx:
            continue
        W, H = S[k]["img_size"]
        if progs[k]["op"] == "along":
            side = int(min(max(side3(k), 1.25 * max(bx[2] - bx[0], bx[3] - bx[1])), min(W, H)))
            b = max(1.0, a.retry_b3 * (side3(k) / side) ** 2)
        else:
            side, b = side3(k), a.retry_b3
        boxB[k] = BoxQSpec(S[k], spec.q, b, box=crop_box(W, H, (bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2, side))
    rB = run_pass(runner, ptB, a.chunk) if ptB else {}
    bB = run_boxq(runner, boxB, a.chunk, 64) if boxB else {}
    newpt, extra_tok, info = {}, collections.Counter(), {}
    for k, p in progs.items():
        if p["op"] == "cell":
            c = (rB[f"{k}||col"]["point"] if f"{k}||col" in rB and rB[f"{k}||col"]["point"] else rA[f"{k}||col"]["point"])
            r = (rB[f"{k}||row"]["point"] if f"{k}||row" in rB and rB[f"{k}||row"]["point"] else rA[f"{k}||row"]["point"])
            for key in (f"{k}||col", f"{k}||row"):
                extra_tok[k] += rA[key]["img_tokens"] + (rB[key]["img_tokens"] if key in rB else 0)
            geo = bool(c and r and c[0] > r[0] and c[1] < r[1])   # column header must be right of and above the row header
            if geo:
                newpt[k] = (c[0], r[1])
            info[k] = dict(col_pt=c, row_pt=r, geo_ok=geo)
            # raw answer, parse status and stated size of the four header passes
            info[k]["passes"] = {f"{st}_{h}": audit_rec(rr.get(f"{k}||{h}")) for st, rr in (("A", rA), ("B", rB))
                                 for h in ("col", "row")}
        elif p["op"] in ("along", "text_edge") and k in bA:
            extra_tok[k] += bA[k]["img_tokens"] + (bB[k]["img_tokens"] if k in bB else 0)
            bx = (bB[k]["boxes"][0] if k in bB and bB[k]["boxes"] else None) or (bA[k]["boxes"][0] if bA[k]["boxes"] else None)
            info[k] = dict(boxA=bA[k]["boxes"][:1], boxB=bB[k]["boxes"][:1] if k in bB else None, rawB=bB[k]["raw"][:100] if k in bB else None)
            if bx:
                if p["op"] == "along":
                    newpt[k] = (bx[0] + p["fx"] * (bx[2] - bx[0]), bx[1] + p["fy"] * (bx[3] - bx[1]))
                else:
                    cy = (bx[1] + bx[3]) / 2
                    newpt[k] = (bx[2] + 1, cy) if p["side"] == "after" else (bx[0] - 1, cy)
    stem = a.output[:-5] if a.output.endswith(".json") else a.output
    for tag, use in (("all", {"cell", "along", "text_edge"}), ("cell", {"cell"}), ("along", {"along"}), ("text", {"text_edge"})):
        p1, meta_r = {}, {k: dict(v) for k, v in meta.items()}
        for s in samples:
            k = s["id"]
            pt, tok = R[k].get("final"), R[k]["vis_tokens"]
            if progs[k]["op"] in use:
                tok += extra_tok[k]
                if k in newpt:
                    pt = newpt[k]
            p1[k] = dict(point=tuple(pt) if pt else None, raw="", img_tokens=tok, prompt_tokens=0, gen_tokens=0,
                         view_size=None, ref_meta=None)
            meta_r[k]["program"] = dict(prog=progs[k], exec=info.get(k), base=R[k].get("final"))
            meta_r[k]["crop"] = R[k].get("crop")
        finalize(a, "program", samples, meta_r, p1, {}, t0, timings, f"{stem}_{tag}.json",
                 extra=dict(program_from=a.retry_from, ops=dict(ops), applied=sorted(use), b3=a.retry_b3))


# ----------------------------------------------------------------------------- TNZ chain (tsn)
def run_tsn(a, runner, samples, meta, t0, timings):
    """Target-scale normalisation: the zoom geometry is derived from the model's token geometry. One visual token covers
    --tsn_patch original pixels, so a target whose shorter side is s px spans s/patch tokens. A zoom is issued while the
    target spans fewer than --tsn_tok_target tokens in the view the model last saw (--tsn_loose x that at the first
    zoom); its up-sampling is
        k = clip(tsn_tok_target * patch / s, 1, --tsn_kmax),
    its crop side
        L = clip(max(--tsn_ctx * s, tsn_err_tokens * patch / applied scale, --retry_min_side), <= min(W, H)),
    which also covers the localisation error of the previous view. After each zoom the box is re-estimated in the new
    view, so the depth (0 .. --tsn_depth) is decided per sample. A zoom is accepted only if its Yes/No verification
    (close-up of side --mv_vside x d at budget --retry_vb) is not worse."""
    S = {s["id"]: s for s in samples}
    sids = list(S)
    patch = a.tsn_patch
    auto_b = {}
    for i in sids:
        W, H = S[i]["img_size"]
        auto_b[i] = min(1.0, max(a.ad_b1_min, a.auto_px / float(W * H)))
    p1 = run_pass(runner, {i: JobSpec(S[i], "shot", auto_b[i]) for i in sids}, a.chunk)
    bq = run_boxq(runner, {i: BoxQSpec(S[i], BOX_Q.format(instr=S[i]["instruction"]), auto_b[i]) for i in sids}, a.chunk, 64)
    cur = {i: p1[i]["point"] for i in sids}
    short = {}
    for i in sids:
        bx = bq[i]["boxes"][0] if bq[i]["boxes"] else None
        short[i] = min(bx[2] - bx[0], bx[3] - bx[1]) if bx else 0.0
    tok = {i: p1[i]["img_tokens"] + bq[i]["img_tokens"] for i in sids}
    applied = {i: auto_b[i] ** 0.5 for i in sids}   # view scale relative to the original image (coarse pass may down-sample)
    info = {i: dict(b1=round(auto_b[i], 3), short_side=round(short[i], 1), stages=[]) for i in sids}
    for i in sids:                                   # the coarse pass's raw answer, parse status, stated size
        info[i]["coarse"] = audit_rec(p1[i])
    vcur = {}
    for depth in range(1, a.tsn_depth + 1):
        plan = {}
        for i in sids:
            if not cur[i] or short[i] <= 0:
                continue
            span = short[i] * applied[i] / patch      # apparent span in the view the model last saw
            need = a.tsn_tok_target * (a.tsn_loose if depth == 1 else 1.0)   # asymmetric cost: be generous about the FIRST zoom
            if span >= need:
                continue
            if depth > 1 and applied[i] >= a.tsn_kmax - 1e-6:
                continue                               # already at maximum magnification: more depth cannot add resolution
            k = min(max(a.tsn_tok_target * patch / short[i], 1.0), a.tsn_kmax)
            W, H = S[i]["img_size"]
            # the crop must also cover the localisation error of the view the point came from: one token of that view is
            # patch / applied original pixels, so an error of tsn_err_tokens tokens is that many pixels.
            err = a.tsn_err_tokens * patch / max(applied[i], 1e-6)
            L = int(min(max(a.tsn_ctx * short[i], err, a.retry_min_side), min(W, H)))   # crop = ctx x target
            plan[i] = (k, L)
        if not plan:
            break
        need_v = [i for i in plan if i not in vcur and cur[i]]      # verify the current answer only where a zoom will happen
        if need_v:
            vnew0 = run_verify(runner, {i: VerifySpec(S[i], cur[i], int(a.mv_vside * meta[i]["d"]), a.retry_vb)
                                        for i in need_v}, a.chunk)
            for i in vnew0:
                vcur[i] = vnew0[i]; tok[i] += vnew0[i]["img_tokens"]
        specs = {}
        for i, (k, L) in plan.items():
            W, H = S[i]["img_size"]
            specs[i] = JobSpec(S[i], "crop", k * k, box=crop_box(W, H, cur[i][0], cur[i][1], L))
        pz = run_pass(runner, specs, a.chunk)
        bz = {} if depth >= a.tsn_depth else run_boxq(   # the last stage needs no size re-estimate: nothing follows it
            runner, {i: BoxQSpec(S[i], BOX_Q.format(instr=S[i]["instruction"]), plan[i][0] ** 2,
                                 box=crop_box(S[i]["img_size"][0], S[i]["img_size"][1], cur[i][0], cur[i][1], plan[i][1]))
                     for i in plan}, a.chunk, 64)
        cand = {i: (pz.get(i) or {}).get("point") for i in plan}
        vz = run_verify(runner, {i: VerifySpec(S[i], cand[i], int(a.mv_vside * meta[i]["d"]), a.retry_vb)
                                 for i in plan if cand[i]}, a.chunk)
        for i in plan:
            k, L = plan[i]
            tok[i] += (pz[i]["img_tokens"] if i in pz else 0) + (bz[i]["img_tokens"] if i in bz else 0) + \
                      (vz[i]["img_tokens"] if i in vz else 0)
            take = bool(cand.get(i)) and (i not in vz or i not in vcur or vz[i]["p_yes"] >= vcur[i]["p_yes"])
            info[i]["stages"].append(dict(depth=depth, k=round(k, 2), side=L, taken=bool(take), span_before=round(short[i] * applied[i] / patch, 2),
                                          p_old=round(vcur[i]["p_yes"], 4) if i in vcur else None,
                                          p_new=round(vz[i]["p_yes"], 4) if i in vz else None))
            if i in pz and "raw" in pz[i]:
                info[i]["stages"][-1]["zoom"] = audit_rec(pz[i])
            if take:
                cur[i] = cand[i]
                applied[i] = k
                if i in vz:
                    vcur[i] = vz[i]
            nb = bz[i]["boxes"][0] if (i in bz and bz[i]["boxes"]) else None
            if nb:
                short[i] = min(nb[2] - nb[0], nb[3] - nb[1])
                info[i]["stages"][-1]["short_after"] = round(short[i], 1)
    p1out, meta_r = {}, {k: dict(v) for k, v in meta.items()}
    for s in samples:
        i = s["id"]
        meta_r[i]["tsn"] = info[i]
        p1out[i] = dict(point=tuple(cur[i]) if cur[i] else None, raw="", img_tokens=tok[i], prompt_tokens=0,
                        gen_tokens=0, view_size=None, ref_meta=None)
    # tsn_view_tokens: kept in the summary for format compatibility (the crop side is set by --tsn_ctx)
    finalize(a, "tsn", samples, meta_r, p1out, {}, t0, timings, a.output,
             extra=dict(tsn_patch=patch, tsn_tok_target=a.tsn_tok_target, tsn_view_tokens=1024.0, tsn_kmax=a.tsn_kmax,
                        tsn_depth=a.tsn_depth, auto_px=a.auto_px, vb=a.retry_vb))


# ----------------------------------------------------------------------------- re-grounding (retry)
EXCLUDE_NOTE = ("\nNote: a previous attempt clicked the location marked with the red X, and that was judged NOT to be the "
                "target. Find the correct element (it may be close to or far from the X).")


def run_retry(a, runner, samples, meta, t0, timings):
    """Verification-triggered re-grounding with negative feedback.
    1) Yes/No-verify the FINAL point of --retry_from (close-up of side --mv_vside x d at budget --retry_vb);
    2) re-run pass 1 (budget --b1) with the rejected point marked by a red X and an exclusion note;  3) native d second
    view;  4) third view of side max(0.5 d, --retry_min_side) up-sampled by --retry_b3;  5) verify the new final.
    The new answer is computed for every sample; one output per tau in --retry_taus (triggered = old p_yes < tau; the
    new answer is taken iff its p_yes is higher).
    Tokens: old pipeline + old verification (always) + re-grounding passes and new verification when triggered."""
    R = {d["id"]: d for d in json.load(open(a.retry_from))["details"]}
    sids = [s["id"] for s in samples if R[s["id"]].get("final")]
    S = {s["id"]: s for s in samples}
    vb = a.retry_vb
    v_old = run_verify(runner, {i: VerifySpec(S[i], R[i]["final"], int(a.mv_vside * meta[i]["d"]), vb) for i in sids}, a.chunk)
    specs1 = {i: JobSpec(S[i], "shot", a.b1, exclude=R[i]["final"], inst=S[i]["instruction"] + EXCLUDE_NOTE) for i in sids}
    p1n = run_pass(runner, specs1, a.chunk)
    specs2 = {}
    for i in sids:
        pt = p1n[i]["point"]
        if pt is None:
            continue
        W, H = S[i]["img_size"]
        specs2[i] = JobSpec(S[i], "crop", 1.0, box=crop_box(W, H, pt[0], pt[1], meta[i]["d"]))
    p2n = run_pass(runner, specs2, a.chunk)
    specs3 = {}
    for i in sids:
        pt = (p2n.get(i) or {}).get("point") or p1n[i]["point"]
        if pt is None:
            continue
        W, H = S[i]["img_size"]
        side = max(int(0.5 * meta[i]["d"]), a.retry_min_side)
        specs3[i] = JobSpec(S[i], "crop", a.retry_b3, box=crop_box(W, H, pt[0], pt[1], side))
    p3n = run_pass(runner, specs3, a.chunk)
    newf = {}
    for i in sids:
        newf[i] = (p3n.get(i) or {}).get("point") or (p2n.get(i) or {}).get("point") or p1n[i]["point"]
    v_new = run_verify(runner, {i: VerifySpec(S[i], newf[i], int(a.mv_vside * meta[i]["d"]), vb) for i in sids if newf[i]}, a.chunk)
    stem = a.output[:-5] if a.output.endswith(".json") else a.output
    for tau in [float(x) for x in a.retry_taus.split(",")]:
        p1, meta_r = {}, {k: dict(v) for k, v in meta.items()}
        for s in samples:
            i = s["id"]
            old = R[i]
            tok = old["vis_tokens"]
            pt = old.get("final")
            info = None
            if i in v_old:
                po = v_old[i]["p_yes"]
                tok += v_old[i]["img_tokens"]
                trig = po < tau
                pn = v_new[i]["p_yes"] if i in v_new else None
                if trig:
                    tok += p1n[i]["img_tokens"] + (p2n[i]["img_tokens"] if i in p2n else 0) + (p3n[i]["img_tokens"] if i in p3n else 0)
                    tok += v_new[i]["img_tokens"] if i in v_new else 0
                    if pn is not None and pn > po and newf[i]:
                        pt = newf[i]
                info = dict(p_old=round(po, 4), p_new=round(pn, 4) if pn is not None else None, triggered=trig,
                            new_final=[round(v, 1) for v in newf[i]] if newf[i] else None,
                            new_p1=[round(v, 1) for v in p1n[i]["point"]] if p1n[i]["point"] else None)
                # raw answer, parse status and stated size of the three re-grounding passes
                info["passes"] = [audit_rec(r.get(i)) for r in (p1n, p2n, p3n)]
            p1[i] = dict(point=tuple(pt) if pt else None, raw="", img_tokens=tok, prompt_tokens=0, gen_tokens=0,
                         view_size=None, ref_meta=None)
            meta_r[i]["retry"] = info
            meta_r[i]["crop"] = old.get("crop")
        finalize(a, "retry", samples, meta_r, p1, {}, t0, timings, f"{stem}_tau{tau}.json",
                 extra=dict(retry_from=a.retry_from, tau=tau, retry_b1=a.b1, retry_min_side=a.retry_min_side, retry_b3=a.retry_b3,
                            retry_mask=False, retry_vb=a.retry_vb))


# ----------------------------------------------------------------------------- main
def run(a):
    global CAP_PIXELS
    CAP_PIXELS = int(a.max_pixels)
    samples = LOADERS[a.dataset](a.data_dir)
    if a.ids_file:  # restrict to a sample-id subset (json list, or dict whose keys are ids)
        keep = set(json.load(open(a.ids_file)))
        samples = [s for s in samples if s["id"] in keep]
    logger.info(f"mode={a.mode} dataset={a.dataset} n={len(samples)} model={a.model_path} "
                f"b1={a.b1} b2={a.b2} eta={a.eta} d=[{a.dmin},{a.dmax}]")
    runner = Runner(a)

    for s in samples:  # sizes only (no decode)
        with Image.open(s["img_path"]) as im:
            W, H = im.size
        s["img_size"] = [W, H]
        b = s["bbox"]
        s["area_frac"] = max(0.0, (b[2] - b[0]) * (b[3] - b[1])) / float(W * H)

    t0 = time.time()
    timings = {}
    mode = a.mode

    # ---------------- per-sample window side d and the whole-screen pass-1 specs ----------------
    specs1, meta = {}, {}
    for s in samples:
        W, H = s["img_size"]
        d = adaptive_d(W, H, a.eta, a.dmin, a.dmax)
        m = dict(d=d)
        specs1[s["id"]] = JobSpec(s, "shot", a.b1)
        meta[s["id"]] = m
    if mode == "choose2":
        run_choose2(a, runner, samples, meta, t0, timings)
        return
    if mode == "retry":
        run_retry(a, runner, samples, meta, t0, timings)
        return
    if mode == "exist":
        run_exist(a, runner, samples)
        return
    if mode == "program":
        run_program(a, runner, samples, meta, t0, timings)
        return
    if mode == "tsn":
        run_tsn(a, runner, samples, meta, t0, timings)
        return

    # ---------------- zoom: pass 1 (whole screen) ----------------
    ta = time.time()
    p1 = run_pass(runner, specs1, a.chunk)
    timings["pass1_s"] = time.time() - ta
    del specs1

    # ---------------- zoom: pass 2 (--crop_wh window around the pass-1 point) ----------------
    p2 = {}
    specs2 = {}
    for s in samples:
        pt = p1[s["id"]]["point"]
        if pt is None:
            continue
        W, H = s["img_size"]
        box = p2_box(a, W, H, pt[0], pt[1])
        meta[s["id"]]["crop"] = box
        specs2[s["id"]] = JobSpec(s, "crop", a.b2, box=box)
    ta = time.time()
    p2 = run_pass(runner, specs2, a.chunk)
    timings["pass2_s"] = time.time() - ta

    finalize(a, mode, samples, meta, p1, p2, t0, timings, a.output)


def finalize(a, mode, samples, meta, p1, p2, t0, timings, output, extra=None):
    """Writes one result file: per-sample records (prediction `final`, visual tokens, stage records, correctness against
    the ground-truth box) and a summary with accuracy per task, UI type and target size."""
    # ---------------- assemble ----------------
    details = []
    for s in samples:
        m = meta[s["id"]]
        r1 = p1[s["id"]]
        rec = dict(id=s["id"], task=s["task"], app=s["app"], ui_type=s["ui_type"], group=s["group"],
                   gt=s["bbox"], img_size=s["img_size"], area_frac=round(s["area_frac"], 7), d=m["d"],
                   p1=[round(v, 1) for v in r1["point"]] if r1["point"] else None,
                   view1=r1["view_size"], tok1=r1["img_tokens"], prompt_tok1=r1["prompt_tokens"],
                   gen_tok1=r1["gen_tokens"], raw1=r1["raw"], refs=r1["ref_meta"], crop=m.get("crop"))
        for k in ("parse", "stated"):          # locator passes only: parse1 = ok | fail, stated1
            if k in r1:
                rec[k + "1"] = r1[k]
        final = r1["point"]
        if mode == "choose2":
            rec["choose"] = m.get("choose")
        if mode == "retry":
            rec["retry"] = m.get("retry")
        if mode == "program":
            rec["program"] = m.get("program")
        if mode == "tsn":
            rec["tsn"] = m.get("tsn")
        if mode == "zoom":
            rec["e1"] = contains(m["crop"], s["bbox"]) if m.get("crop") else False
            if s["id"] in p2:
                r2 = p2[s["id"]]
                rec.update(p2=[round(v, 1) for v in r2["point"]] if r2["point"] else None,
                           view2=r2.get("view_size"), tok2=r2["img_tokens"], prompt_tok2=r2["prompt_tokens"],
                           gen_tok2=r2["gen_tokens"], raw2=r2["raw"])
                for k in ("parse", "stated"):  # parse2, stated2
                    if k in r2:
                        rec[k + "2"] = r2[k]
                if r2["point"] is not None:
                    final = r2["point"]
            else:
                rec.update(p2=None, tok2=0, prompt_tok2=0, gen_tok2=0, raw2="")
        rec["final"] = [round(v, 1) for v in final] if final else None
        rec["correct_p1"] = in_bbox(r1["point"], s["bbox"])
        rec["correct"] = in_bbox(final, s["bbox"])
        rec["vis_tokens"] = rec["tok1"] + rec.get("tok2", 0)
        rec["gen_tokens"] = rec["gen_tok1"] + rec.get("gen_tok2", 0)
        details.append(rec)

    n = len(details)
    acc = 100.0 * sum(d["correct"] for d in details) / max(n, 1)
    per = {}
    for key in ("task", "ui_type"):
        per[key] = {}
        for d in details:
            per[key].setdefault(d[key], [0, 0])
            per[key][d[key]][1] += 1
            per[key][d[key]][0] += bool(d["correct"])
        per[key] = {k: dict(acc=round(100.0 * c / t, 2), correct=c, total=t) for k, (c, t) in sorted(per[key].items())}
    bins = [("tiny", 0, 1e-4), ("small", 1e-4, 5e-4), ("medium", 5e-4, 2e-3), ("large", 2e-3, 1.0)]
    per["size"] = {}
    for name, lo, hi in bins:
        sub = [d for d in details if lo <= d["area_frac"] < hi]
        per["size"][name] = dict(acc=round(100.0 * sum(d["correct"] for d in sub) / max(len(sub), 1), 2),
                                 correct=sum(d["correct"] for d in sub), total=len(sub))
    mean_vis = sum(d["vis_tokens"] for d in details) / max(n, 1)
    e1 = [d["e1"] for d in details if "e1" in d]
    # mem_*, window (except crop_wh), prompt, p1_from, p1_use_final, eval_scope and n_predicted are constants kept for
    # format compatibility with earlier result files
    summary = dict(mode=mode, dataset=a.dataset, model=a.model_path, family=a.family, b1=a.b1, b2=a.b2,
                   eta=a.eta, dmin=a.dmin, dmax=a.dmax, mem_dir=None, mem_k=3,
                   mem_control="retrieved", mem_alpha=0.5, n=n,
                   n_correct=sum(d["correct"] for d in details),
                   overall_accuracy=round(acc, 2), mean_vis_tokens=round(mean_vis, 1),
                   mean_tok1=round(sum(d["tok1"] for d in details) / max(n, 1), 1),
                   mean_tok2=round(sum(d.get("tok2", 0) for d in details) / max(n, 1), 1),
                   mean_gen_tokens=round(sum(d["gen_tokens"] for d in details) / max(n, 1), 2),
                   e1_rate=round(100.0 * sum(e1) / len(e1), 2) if e1 else None,
                   parse_fail_p1=sum(d["p1"] is None for d in details),
                   elapsed_s=round(time.time() - t0, 1), timings={k: round(v, 1) for k, v in timings.items()},
                   per=per, window=dict(crop_wh=a.crop_wh, crop_scale=1.0, crop_min_side=0, p2_kind="crop"),
                   prompt=dict(inst_mode="none"), p1_from=None, p1_use_final=False,
                   factor=a.factor, max_pixels=CAP_PIXELS,
                   eval_scope="all", n_predicted=n, **(extra or {}))
    logger.info("=" * 70)
    logger.info(f"MODE={mode} b1={a.b1} b2={a.b2} ACC {acc:.2f}% ({summary['n_correct']}/{n}) "
                f"vis_tokens/sample {mean_vis:.0f} (p1 {summary['mean_tok1']:.0f} + p2 {summary['mean_tok2']:.0f}) "
                f"E1 {summary['e1_rate']}  time {summary['elapsed_s']}s")
    logger.info("=" * 70)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    json.dump(dict(summary=summary, details=details), open(output, "w"), indent=1)
    logger.info(f"saved {output}")


def main():
    p = argparse.ArgumentParser(description="AdaGround inference stages for Qwen3-VL backbones (vLLM).")
    p.add_argument("--mode", required=True, choices=["zoom", "exist", "program", "tsn", "choose2", "retry"],
                   help="zoom = fixed 4x crop chain, exist = existence query, program = CSG, tsn = TNZ chain, "
                        "choose2 = arbitration, retry = re-grounding")
    p.add_argument("--dataset", required=True, choices=list(LOADERS))
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--family", default="qwen3", choices=["qwen3"], help="backbone family (Qwen3-VL)")
    p.add_argument("--factor", type=int, default=32, help="resize grid of the views; must equal the processor's "
                                                          "patch_size x merge_size (32 for Qwen3-VL)")
    p.add_argument("--max_pixels", type=float, default=8847360, help="pixel cap of every view (4096*2160)")
    p.add_argument("--output", required=True)
    p.add_argument("--ids_file", default=None, help="json list/dict of sample ids to evaluate (subset)")
    p.add_argument("--b1", type=float, default=1.0, help="whole-screen budget (fraction of the full-resolution pixel count)")
    p.add_argument("--b2", type=float, default=1.0, help="zoom pass-2 / arbitration verification budget (up-sampling of the "
                                                         "pixel count)")
    p.add_argument("--eta", type=float, default=8.0, help="adaptive window side d = clip(sqrt(W*H/eta), dmin, dmax)")
    p.add_argument("--dmin", type=int, default=512)
    p.add_argument("--dmax", type=int, default=1024)
    p.add_argument("--crop_wh", default="", help="zoom pass 2: fixed W,H window (e.g. 1288,728)")
    p.add_argument("--mv_tau", type=float, default=14.0, help="choose2: grouping threshold in original px")
    p.add_argument("--mv_k", type=int, default=3, help="choose2: candidates (largest groups) sent to verification")
    p.add_argument("--mv_vside", type=float, default=1.0, help="verification crop side as a fraction of d")
    p.add_argument("--choose_from", default="", help="choose2: comma list of result files whose FINAL points compete")
    p.add_argument("--choose_method", default="verify", choices=["verify"], help="choose2: verification arbitration")
    p.add_argument("--choose_tok", default="sum", choices=["sum"], help="choose2: visual tokens of all input files are summed")
    p.add_argument("--choose_default", default="first", choices=["first"],
                   help="choose2: without arbitration the first file's point is kept")
    p.add_argument("--retry_from", default="", help="retry/program: result file whose FINAL points are re-grounded / "
                                                    "the base answers")
    p.add_argument("--retry_taus", default="0.99", help="retry: comma list of trigger thresholds tau, one output each")
    p.add_argument("--retry_min_side", type=int, default=192, help="minimum side of the tight views (retry, program, tsn)")
    p.add_argument("--retry_b3", type=float, default=9.0, help="retry/program: up-sampling budget of the tight view")
    p.add_argument("--retry_vb", type=float, default=4.0, help="retry/tsn: verification crop budget")
    p.add_argument("--tsn_patch", type=int, default=32, help="tsn: original pixels covered by one visual token "
                                                            "(Qwen3-VL: 16 x merge 2)")
    p.add_argument("--tsn_tok_target", type=float, default=3.0, help="tsn: visual tokens the target must span (tau)")
    p.add_argument("--tsn_kmax", type=float, default=4.0, help="tsn: maximum up-sampling")
    p.add_argument("--tsn_err_tokens", type=float, default=4.0, help="tsn: localisation error budget of the previous view, "
                                                                     "in ITS tokens")
    p.add_argument("--tsn_ctx", type=float, default=16.0, help="tsn: crop side = ctx x predicted target short side")
    p.add_argument("--tsn_loose", type=float, default=2.0, help="tsn: the first zoom fires when the span is below loose x "
                                                               "tok_target (asymmetric cost)")
    p.add_argument("--tsn_depth", type=int, default=2, help="tsn: maximum zoom depth")
    p.add_argument("--auto_px", type=float, default=2.2e6, help="tsn: target pixel count of the coarse whole-screen view")
    p.add_argument("--ad_b1_min", type=float, default=0.25, help="tsn: lower bound of the coarse-pass budget")
    p.add_argument("--chunk", type=int, default=32, help="samples per vLLM generate call")
    p.add_argument("--max_num_seqs", type=int, default=8)
    p.add_argument("--max_model_len", type=int, default=32768)
    p.add_argument("--gpu_mem_util", type=float, default=0.40)
    p.add_argument("--tp", type=int, default=1)
    a = p.parse_args()
    if a.mode == "zoom" and not a.crop_wh:
        p.error("--mode zoom needs --crop_wh W,H")
    if a.tsn_ctx <= 0:
        p.error("--tsn_ctx must be positive")
    run(a)


if __name__ == "__main__":
    main()
