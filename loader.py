import numpy as np
from PIL import Image
from huggingface_hub import snapshot_download
from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.garment_agnostic_mask_predictor import AutoMasker
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center, list_dir, get_agnostic_mask_hd, get_agnostic_mask_dc
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose


# Download checkpoints
snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts")


class LeffaPredictor(object):
    def __init__(self):
        self.mask_predictor = AutoMasker(
            densepose_path="./ckpts/densepose",
            schp_path="./ckpts/schp",
        )

        self.densepose_predictor = DensePosePredictor(
            config_path="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml",
            weights_path="./ckpts/densepose/model_final_162be9.pkl",
        )

        self.parsing = Parsing(
            atr_path="./ckpts/humanparsing/parsing_atr.onnx",
            lip_path="./ckpts/humanparsing/parsing_lip.onnx",
        )

        self.openpose = OpenPose(
            body_model_path="./ckpts/openpose/body_pose_model.pth",
        )

        vt_model_hd = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon.pth",
        )
        self.vt_inference_hd = LeffaInference(model=vt_model_hd)

        vt_model_dc = LeffaModel(
            pretrained_model_name_or_path="./ckpts/stable-diffusion-inpainting",
            pretrained_model="./ckpts/virtual_tryon_dc.pth",
        )
        self.vt_inference_dc = LeffaInference(model=vt_model_dc)

    def leffa_predict(
        self,
        src_image_path,
        ref_image_path,
        control_type,
        ref_acceleration=True,
        step=50,
        scale=2.5,
        seed=42,
        vt_model_type="viton_hd",
        vt_garment_type="upper_body",
        vt_repaint=False
    ):
        assert control_type == "virtual_tryon", "Invalid control type: {}".format(control_type)
        src_image = Image.open(src_image_path)
        ref_image = Image.open(ref_image_path)
        src_image = resize_and_center(src_image, 768, 1024)
        ref_image = resize_and_center(ref_image, 768, 1024)

        src_image_array = np.array(src_image)

        src_image = src_image.convert("RGB")
        model_parse, _ = self.parsing(src_image.resize((384, 512)))
        keypoints = self.openpose(src_image.resize((384, 512)))
        if vt_model_type == "viton_hd":
            mask = get_agnostic_mask_hd(
                model_parse, keypoints, vt_garment_type)
        elif vt_model_type == "dress_code":
            mask = get_agnostic_mask_dc(
                model_parse, keypoints, vt_garment_type)
        mask = mask.resize((768, 1024))

        # DensePose
        if vt_model_type == "viton_hd":
            src_image_seg_array = self.densepose_predictor.predict_seg(
                src_image_array)[:, :, ::-1]
            src_image_seg = Image.fromarray(src_image_seg_array)
            densepose = src_image_seg
        elif vt_model_type == "dress_code":
            src_image_iuv_array = self.densepose_predictor.predict_iuv(
                src_image_array)
            src_image_seg_array = src_image_iuv_array[:, :, 0:1]
            src_image_seg_array = np.concatenate(
                [src_image_seg_array] * 3, axis=-1)
            src_image_seg = Image.fromarray(src_image_seg_array)
            densepose = src_image_seg

        # Leffa
        transform = LeffaTransform()

        data = {
            "src_image": [src_image],
            "ref_image": [ref_image],
            "mask": [mask],
            "densepose": [densepose],
        }
        data = transform(data)
        if vt_model_type == "viton_hd":
            inference = self.vt_inference_hd
        elif vt_model_type == "dress_code":
            inference = self.vt_inference_dc

        output = inference(
            data,
            ref_acceleration=ref_acceleration,
            num_inference_steps=step,
            guidance_scale=scale,
            seed=seed,
            repaint=vt_repaint,)
        gen_image = output["generated_image"][0]
        # gen_image.save("gen_image.png")
        return gen_image

    def leffa_predict_vt(self, src_image_path, ref_image_path, ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint):
         return self.leffa_predict(src_image_path, ref_image_path, "virtual_tryon", ref_acceleration, step, scale, seed, vt_model_type, vt_garment_type, vt_repaint)

# Load model instance
leffa_predictor = None


def predict_image(src_image_path, ref_image_path, vt_ref_acceleration, vt_step, vt_scale, vt_seed, vt_model_type, vt_garment_type, vt_repaint):
    global leffa_predictor
    if leffa_predictor is None:
        leffa_predictor = LeffaPredictor() #Initialize only when needed.
    return leffa_predictor.leffa_predict_vt(
            src_image_path, ref_image_path, vt_ref_acceleration, vt_step, vt_scale, vt_seed, vt_model_type, vt_garment_type, vt_repaint
        )