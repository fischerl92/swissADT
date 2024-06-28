import torch
import json
import logging
import os

# https://github.com/wjun0830/CGDETR.git
# 99f110841615b786498d4a9f87afd5d665cd185f

from .run_on_video.data_utils import ClipFeatureExtractor
from .run_on_video.model_utils import build_inference_model
from .utils.tensor_utils import pad_sequences_1d
from .cg_detr.span_utils import span_cxw_to_xx
from moviepy.video.io.VideoFileClip import VideoFileClip
import torch.nn.functional as F

from swiss_adt import save_subclip


class CGDETRPredictor:
    def __init__(
        self, ckpt_path=None, clip_model_name_or_path="ViT-B/32", device="cuda"
    ):
        if ckpt_path is None:
            ckpt_path = os.path.join(
                os.path.dirname(__file__), "qvhighlights_onlyCLIP.ckpt"
            )
        self.clip_len = 2  # seconds
        self.device = device
        logging.info("Loading feature extractors...")
        self.feature_extractor = ClipFeatureExtractor(
            framerate=1 / self.clip_len,
            size=224,
            centercrop=True,
            model_name_or_path=clip_model_name_or_path,
            device=device,
        )
        logging.info("Loading trained CG-DETR model...")
        self.model = build_inference_model(ckpt_path).to(self.device)

    @torch.no_grad()
    def localize_moment(self, video_path, query_list):
        """
        Args:
            video_path: str, path to the video file
            query_list: List[str], each str is a query for this video
        """
        # construct model inputs
        n_query = len(query_list)
        video_feats = self.feature_extractor.encode_video(video_path)
        video_feats = F.normalize(video_feats, dim=-1, eps=1e-5)
        n_frames = len(video_feats)
        # add tef
        tef_st = torch.arange(0, n_frames, 1.0) / n_frames
        tef_ed = tef_st + 1.0 / n_frames
        tef = torch.stack([tef_st, tef_ed], dim=1).to(self.device)  # (n_frames, 2)
        video_feats = torch.cat([video_feats, tef], dim=1)
        assert n_frames <= 75, (
            "The positional embedding of this pretrained CGDETR only support video up "
            "to 150 secs (i.e., 75 2-sec clips) in length"
        )
        video_feats = video_feats.unsqueeze(0).repeat(n_query, 1, 1)  # (#text, T, d)
        video_mask = torch.ones(n_query, n_frames).to(self.device)
        query_feats = self.feature_extractor.encode_text(query_list)  # #text * (L, d)
        query_feats, query_mask = pad_sequences_1d(
            query_feats, dtype=torch.float32, device=self.device, fixed_length=None
        )
        query_feats = F.normalize(query_feats, dim=-1, eps=1e-5)
        model_inputs = dict(
            src_vid=video_feats,
            src_vid_mask=video_mask,
            src_txt=query_feats,
            src_txt_mask=query_mask,
            vid=None,
            qid=None,
        )

        # decode outputs
        outputs = self.model(**model_inputs)
        # #moment_queries refers to the positional embeddings in CGDETR's decoder, not the input text query
        prob = F.softmax(
            outputs["pred_logits"], -1
        )  # (batch_size, #moment_queries=10, #classes=2)
        scores = prob[
            ..., 0
        ]  # * (batch_size, #moment_queries)  foreground label is 0, we directly take it
        pred_spans = outputs["pred_spans"]  # (bsz, #moment_queries, 2)

        # compose predictions
        predictions = []
        video_duration = n_frames * self.clip_len
        for idx, (spans, score) in enumerate(zip(pred_spans.cpu(), scores.cpu())):
            spans = span_cxw_to_xx(spans) * video_duration
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            cur_ranked_preds = sorted(
                cur_ranked_preds, key=lambda x: x[2], reverse=True
            )
            cur_ranked_preds = [
                [float(f"{e:.4f}") for e in row] for row in cur_ranked_preds
            ]
            cur_query_pred = dict(
                query=query_list[idx],  # str
                vid=video_path,
                pred_relevant_windows=cur_ranked_preds,  # List([st(float), ed(float), score(float)])
            )
            predictions.append(cur_query_pred)

        return predictions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logging.info("Build models...")

    cg_detr_predictor = CGDETRPredictor(
        ckpt_path="qvhighlights_onlyCLIP.ckpt", device="cpu"
    )

    with open("../example/example.jsonl", "r") as file:
        for line in file:
            segment = json.loads(line)
            # get frame from video and encode it
            video_path = segment["video"]
            query_text_list = [segment["audio_description"]]

            logging.info(
                f"For video: {video_path}\n find moment for query: {query_text_list}"
            )
            predictions = cg_detr_predictor.localize_moment(
                video_path=video_path, query_list=query_text_list
            )

            moment = predictions[0]["pred_relevant_windows"][0]
            save_subclip(
                video_path, "../example/moment_ruedi.mp4", moment[0], moment[1]
            )
