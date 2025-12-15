python scripts/mesh_from_mask.py --image notebook/images/kid_box/image.png --mask notebook/images/kid_box/7.png --output-dir artifacts/kid_box

python scripts/viz_mesh.py artifacts/kid_box/scene.glb --output-dir scripts/viz_dir --num-views 6 --num-elevations 6
