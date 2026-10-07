#!/usr/bin/env python3
"""
Convert outputs_avocado/cnn_final.pt into files that run without PyTorch:

  cnn_final.onnx   inputs: image (1x3xHxW, ImageNet normalised) [, cond]
                   outputs: logits (1x5), days_scaled (1), fmap (1xCxhxw)
  cnn_head.npz     weights of the stage head, used to compute the class
                   activation map, plus the metadata of the checkpoint

For a ResNet the stage logit is linear in the global average of the last
feature map, so Grad-CAM reduces to CAM: cam = relu(sum_k W[c, k] * fmap_k).
The script checks that the ONNX outputs match PyTorch before saving.

    python export_onnx.py --models-dir outputs_avocado
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pipeline as P


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.store = {}
        net.cam_layer.register_forward_hook(lambda m, i, o: self.store.__setitem__("fmap", o))

    def forward(self, x, cond):
        logits, reg = self.net(x, cond)
        return logits, reg, self.store["fmap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=Path("outputs_avocado"))
    args = ap.parse_args()
    ckpt = torch.load(args.models_dir / "cnn_final.pt", map_location="cpu", weights_only=False)
    n_cond = len(ckpt["conditions"]) if ckpt.get("use_condition") else 0
    net = P.MultiTaskNet(ckpt["backbone"], len(P.STAGES), n_cond, pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    wrap = Wrapper(net).eval()
    size = ckpt["img_size"]
    x = torch.randn(1, 3, size, size)
    cond = torch.zeros(1, max(n_cond, 1))[:, :n_cond]
    out_path = args.models_dir / "cnn_final.onnx"
    torch.onnx.export(wrap, (x, cond), str(out_path), input_names=["image", "cond"],
                      output_names=["logits", "days_scaled", "fmap"], opset_version=17, dynamo=False)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    feeds = {"image": x.numpy()}
    if n_cond:
        feeds["cond"] = cond.numpy()
    o = sess.run(None, {k: v for k, v in feeds.items() if k in [i.name for i in sess.get_inputs()]})
    with torch.no_grad():
        t = wrap(x, cond)
    for a, b, name in zip(o, t, ["logits", "days", "fmap"]):
        err = float(np.abs(a - b.numpy()).max())
        print(f"[CHECK] {name}: max difference {err:.2e}")
        assert err < 1e-3, name

    W = net.cls_head.weight.detach().numpy()
    meta = {k: ckpt[k] for k in ["backbone", "mode", "y_scale", "img_size", "conditions", "use_condition",
                                 "ready_stage", "crop_pad", "crop_size"] if k in ckpt}
    np.savez(args.models_dir / "cnn_head.npz", W=W, meta=json.dumps(meta))
    print(f"[OK] {out_path} and cnn_head.npz saved")


if __name__ == "__main__":
    main()
