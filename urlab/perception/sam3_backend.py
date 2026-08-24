"""SAM3 segmentation backend -- VENDORED from sam3-abhay/scripts/cable_neck_core.NeckDetector.

WHAT IS AND IS NOT A DEPENDENCY. This class is OUR code about SAM3: which confidence threshold each
prompt gets, the attention-backend workaround that makes the model fit a 6 GB pre-Ampere card, the
decision to autocast rather than half() the weights, and the order the prompts are run in. None of
that is the model, so none of it had any business living in a checkout the test suite cannot see.

The MODEL is a dependency like torch or numpy: `import sam3`, plus its weights. It resolves its own
tokenizer through pkg_resources and its checkpoint from HuggingFace, so it needs to be INSTALLED,
not to be at some path -- which is why sam3.repo_path is gone. If `import sam3` fails, install it
(pip install -e /path/to/sam3-abhay) rather than pointing a config at a directory.

Renamed from NeckDetector: perception.sam3 already exports a class by that name (the adapter that
wraps this one), and two of them in one package is a trap. Everything else is verbatim.
"""

import os

from .neck import compute_necks, compute_tip, masks_from_output, masks_scores_from_output

__all__ = ['Sam3Backend']


class Sam3Backend:
    """SAM3-backed cable-neck detector. Loads the model once; reuse across frames."""

    def __init__(self, cable_prompt="cable", connector_prompt="connector",
                 threshold=0.5, connector_threshold=None, mislabel_overlap=0.6):
        import torch
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._torch = torch
        self.cable_prompt = cable_prompt
        self.connector_prompt = connector_prompt
        self.mislabel_overlap = mislabel_overlap
        # SEPARATE thresholds per prompt. The connector is the harder class: it is small, and SAM3
        # happily labels the whole assembly "cable", which leaves conn_masks EMPTY -- and since
        # compute_necks iterates over CONNECTORS, that yields ZERO necks no matter how good the cable
        # mask is. A lower threshold for the connector alone lets marginal detections through WITHOUT
        # flooding the cable side with spurious masks; one shared threshold cannot do both.
        self.cable_threshold = float(threshold)
        self.connector_threshold = (float(threshold) if connector_threshold is None
                                    else float(connector_threshold))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # SDPA backend -- this is what makes SAM3 fit on a small, pre-Ampere GPU.
        # SAM3's get_sdpa_settings() disables Flash Attention on any GPU below Ampere and falls back
        # to the MATH kernel, which MATERIALIZES the full NxN attention matrix (~822 MB at 1008px)
        # and OOMs a 6 GB card. The cutlass MEMORY-EFFICIENT backend runs on Pascal and never
        # materializes it. It is only eligible for fp16 (bf16 needs sm_80), which is why detect()
        # autocasts to float16 on CUDA. Weights stay fp32 -- autocast casts per op, so there is no
        # fp16/fp32 mismatch (unlike .half()'ing the model, which SAM3 does not support).
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)         # needs Ampere; unavailable here
            torch.backends.cuda.enable_mem_efficient_sdp(True)  # the one that saves the memory
            torch.backends.cuda.enable_math_sdp(True)           # keep only as a last-resort fallback
        model = build_sam3_image_model()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Do NOT .half() this model. SAM3 creates fp32 tensors internally all over its graph (decoder
        # queries, text embeddings, ...), and autocast doesn't cover those paths -- fp16 weights then
        # collide with them ("mat1 and mat2 must have the same dtype, but got Float and Half") in one
        # op after another. Keep fp32 weights + autocast (see detect); to fit a small GPU, free other
        # VRAM (close GPU-accelerated apps) rather than changing the model's precision.
        # RESOLUTION MUST STAY 1008: the ViTDet backbone's RoPE `freqs_cis` buffer is baked for that
        # grid, so any other value trips the assert in vitdet.reshape_for_broadcast.
        resolution = int(os.environ.get("SAM3_RESOLUTION", "1008"))
        self.processor = Sam3Processor(model, resolution=resolution,
                                       confidence_threshold=threshold)

    def detect(self, pil_rgb):
        """Run SAM3 + geometry on a PIL RGB image. Returns the compute_necks dict
        plus the raw cable/connector counts."""
        torch = self._torch
        W, H = pil_rgb.size
        # float16 on CUDA: REQUIRED for the memory-efficient SDPA backend on a pre-Ampere GPU (bf16
        # needs sm_80; without fp16 the attention silently falls back to the math kernel, which
        # materializes the full NxN matrix and OOMs). bfloat16 on CPU.
        ac_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        autocast = torch.autocast(self.device, dtype=ac_dtype)
        with torch.inference_mode(), autocast:
            state = self.processor.set_image(pil_rgb)
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(self.cable_threshold)
            cable_masks = masks_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.cable_prompt))
            self.processor.reset_all_prompts(state)
            # Lower bar for the connector -- see __init__: no connector mask means no neck at all.
            self.processor.set_confidence_threshold(self.connector_threshold)
            conn_masks_raw = masks_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.connector_prompt))
        result = compute_necks(cable_masks, conn_masks_raw, H, W, self.mislabel_overlap)
        result["cables_raw"] = len(cable_masks)
        result["connectors_raw"] = len(conn_masks_raw)
        return result

    def _segment_both(self, pil_rgb):
        """Run BOTH prompts once and return (cable_masks, conn_masks_raw). Shared by detect/detect_tip."""
        torch = self._torch
        ac_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        with torch.inference_mode(), torch.autocast(self.device, dtype=ac_dtype):
            state = self.processor.set_image(pil_rgb)
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(self.cable_threshold)
            cable_masks = masks_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.cable_prompt))
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(self.connector_threshold)
            conn_masks_raw = masks_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.connector_prompt))
        return cable_masks, conn_masks_raw

    def _segment_both_scored(self, pil_rgb, floor):
        """Run BOTH prompts ONCE at the confidence FLOOR; return (cable_masks, cable_scores,
        conn_masks, conn_scores), each sorted by score descending. One inference -- the threshold
        search then happens purely in software."""
        torch = self._torch
        ac_dtype = torch.float16 if self.device == "cuda" else torch.bfloat16
        with torch.inference_mode(), torch.autocast(self.device, dtype=ac_dtype):
            state = self.processor.set_image(pil_rgb)
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(floor)
            cable_m, cable_s = masks_scores_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.cable_prompt))
            self.processor.reset_all_prompts(state)
            self.processor.set_confidence_threshold(floor)
            conn_m, conn_s = masks_scores_from_output(
                self.processor.set_text_prompt(state=state, prompt=self.connector_prompt))
        return cable_m, cable_s, conn_m, conn_s

    def detect_adaptive(self, pil_rgb, floor=0.05, mislabel_overlap=None):
        """ADAPTIVE-THRESHOLD neck detection: tune the thresholds per image, subject to a confidence
        floor, until the cable and connector actually TOUCH (which is what defines the neck).

        The problem with a fixed threshold. SAM3 puts the connector in the "cable" bucket (or the cable
        in the "connector" bucket) depending on the frame. Whichever class comes up empty, compute_necks
        emits nothing -- it needs BOTH, because a neck IS the cable/connector contact. One global
        threshold cannot be right for every frame.

        The fix. Do not commit to a threshold. Get every candidate mask WITH ITS SCORE, then search the
        threshold PAIR, keeping the MOST CONFIDENT masks that still yield a valid neck. The acceptance
        test is GEOMETRIC, not confidence-based -- which is the whole point: a mislabel only matters if
        it destroys the cable/connector contact.

        Why this is nearly free:
          * The threshold is only a FILTER on per-mask scores (`keep = out_probs > threshold`) -- the
            forward pass is identical. So sweeping it costs NO extra inference: one run at `floor`.
          * Only the actual mask SCORES are distinguishable thresholds. Keeping the top-k of a class is
            therefore the complete set of reachable subsets, so the search is exactly
            len(cable) x len(connector) cheap geometry calls -- typically well under 25.

        Preference order: the combination whose LOWEST admitted score is HIGHEST -- i.e. the highest
        thresholds that still produce a neck. Never admits anything below `floor`.

        Returns the compute_necks dict plus: cables_raw, connectors_raw, thr_cable, thr_conn, eff_conf
        (the chosen effective confidence), combos_tried. thr_* are None if nothing worked.
        """
        W, H = pil_rgb.size
        mo = self.mislabel_overlap if mislabel_overlap is None else mislabel_overlap
        cable_m, cable_s, conn_m, conn_s = self._segment_both_scored(pil_rgb, floor)

        # Every reachable (cable, connector) subset pair, ranked by effective confidence.
        cands = []
        for kc in range(1, len(cable_m) + 1):
            for kk in range(1, len(conn_m) + 1):
                eff = min(cable_s[kc - 1], conn_s[kk - 1])   # the weakest mask this pair admits
                if eff < floor:
                    continue
                cands.append((eff, kc, kk))
        cands.sort(key=lambda t: -t[0])                      # most confident first

        for tried, (eff, kc, kk) in enumerate(cands, start=1):
            res = compute_necks(cable_m[:kc], conn_m[:kk], H, W, mo)
            if res["necks"]:                                 # <-- the geometric acceptance test
                res.update(cables_raw=len(cable_m), connectors_raw=len(conn_m),
                           thr_cable=cable_s[kc - 1], thr_conn=conn_s[kk - 1],
                           eff_conf=eff, combos_tried=tried)
                return res

        # No admissible threshold pair yields a neck. Return the most permissive result so the debug
        # overlay still shows what SAM3 actually saw (which is how you diagnose it).
        res = compute_necks(cable_m, conn_m, H, W, mo)
        res.update(cables_raw=len(cable_m), connectors_raw=len(conn_m),
                   thr_cable=None, thr_conn=None, eff_conf=None, combos_tried=len(cands))
        return res

    def detect_tip(self, pil_rgb, curve_px=40):
        """Classification-free CONNECTOR TIP detection: union both prompts, then use the SHAPE.

        Unlike detect() (which needs a connector mask to exist, since compute_necks iterates over
        connectors), this survives SAM3 labelling the whole assembly "cable" -- the common failure. See
        compute_tip. Always returns a dict; res['tip'] is None when nothing usable was found."""
        W, H = pil_rgb.size
        cable_masks, conn_masks_raw = self._segment_both(pil_rgb)
        res = compute_tip(cable_masks, conn_masks_raw, H, W, curve_px=curve_px)
        if res is None:
            res = dict(mask=None, tip=None, back=None, direction=None, angle_deg=0.0,
                       thickness_px=0.0, used_connector=False)
        res["cables_raw"] = len(cable_masks)
        res["connectors_raw"] = len(conn_masks_raw)
        return res
