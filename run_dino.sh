"""
python -m demo.inference_on_a_image \
  -c groundingdino/config/GroundingDINO_SwinB.cfg.py \
  -p https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -i .asset/cats.png \
  -o "outputs/0" \
  -t "cat ear." #\
#   [--cpu-only] # open it for cpu mode
"""

python -m demo.inference_on_a_image \
  -c groundingdino/config/GroundingDINO_SwinT_OGC.py \
  -p groundingdino_swint_ogc.pth \
  -i .asset/cats.png \
  -o "outputs/0" \
  -t "cat ear." #\