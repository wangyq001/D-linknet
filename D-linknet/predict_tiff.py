import torch
import os
import numpy as np
import cv2
from PIL import Image
from time import time
import rasterio
import csv

from networks.dinknet import DinkNet34


INPUT_TIFF = '/root/autodl-tmp/DLinknet/D-linknet/dataset/data/caofg_area2.tif'
WEIGHT_PATH = '/root/autodl-tmp/DLinknet/D-linknet/weights/log01_dink34.th'
OUTPUT_DIR = '/root/autodl-tmp/DLinknet/D-linknet/submits/tiff_inference'

PATCH_SIZE = 1024
STRIDE = 768
BATCHSIZE_PER_CARD = 16 # 必须是8的倍数
MASK_THRESHOLD = 0.5


def normalize_for_model(img):
    """img: uint8 HxWx3 in BGR order (cv2 read). Output: float32 CxHxW normalized to [-1.6, 1.6]."""
    img = img.astype(np.float32) / 255.0 * 3.2 - 1.6
    img = img.transpose(2, 0, 1)
    return img


def predict_tta_batch(net, batch_tensor):
    """TTA: for each tile in the batch, apply 8 augmentations, average, return one result per tile.
    Processes one tile at a time to stay within GPU memory. Returns shape (N, H, W)."""
    net.eval()
    with torch.no_grad():
        N = batch_tensor.shape[0]
        img = batch_tensor.cpu().numpy()

        results = []
        for i in range(N):
            patch = img[i]
            orig = patch
            r90  = np.rot90(patch, 1, axes=(1, 2))
            fh   = patch[:, :, ::-1].copy()
            fv   = patch[:, ::-1, :].copy()
            fhv  = patch[:, ::-1, ::-1].copy()
            r90fh  = np.rot90(patch, 1, axes=(1, 2))[:, :, ::-1].copy()
            r90fv  = np.rot90(patch, 1, axes=(1, 2))[:, ::-1, :].copy()
            r90fhv = np.rot90(patch, 1, axes=(1, 2))[:, ::-1, ::-1].copy()
            aug = np.stack([orig, r90, fh, fv, fhv, r90fh, r90fv, r90fhv], axis=0)
            aug_t = torch.from_numpy(aug).cuda()
            out = net(aug_t).squeeze(1).cpu().numpy()
            avg = (out[0] + out[1] + out[2] + out[3] + out[4] + out[5] + out[6] + out[7]) * 0.125
            results.append(avg)
        return np.stack(results, axis=0)


def mirror_pad(img, patch_size, stride):
    """Pad image so its dimensions are multiples of stride, using mirror reflection.
    This guarantees every sliding-window tile has exactly patch_size shape.
    Returns padded_img and the (top, left) offset of the original image inside the pad."""
    h, w = img.shape[:2]
    # How many full stride-steps fit
    pad_right = max(0, stride - (w % stride)) if w % stride != 0 else 0
    pad_bottom = max(0, stride - (h % stride)) if h % stride != 0 else 0
    # Always pad at least one patch so the final tile fits
    if pad_right == 0 and w % stride == 0 and w > 0:
        pad_right = 0
    if pad_bottom == 0 and h % stride == 0 and h > 0:
        pad_bottom = 0
    # Clamp so we don't overshoot patch_size
    pad_right = min(pad_right, patch_size)
    pad_bottom = min(pad_bottom, patch_size)

    if pad_right > 0 or pad_bottom > 0:
        img = np.pad(img, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='reflect')
    return img, (h, w)


def sliding_window_inference(net, full_img_bgr, patch_size, stride):
    """Sliding window with full-image mirror padding so all tiles are patch_size x patch_size."""
    h_ori, w_ori = full_img_bgr.shape[:2]

    padded, (h_actual, w_actual) = mirror_pad(full_img_bgr, patch_size, stride)
    h_pad, w_pad = padded.shape[:2]

    score_map = np.zeros((h_pad, w_pad), dtype=np.float32)
    weight_map = np.zeros((h_pad, w_pad), dtype=np.float32)

    tiles = []
    for y in range(0, h_pad - patch_size + 1, stride):
        for x in range(0, w_pad - patch_size + 1, stride):
            tiles.append((y, x))

    tta_factor = 8
    n_tiles = len(tiles)
    n_padded = ((n_tiles + tta_factor - 1) // tta_factor) * tta_factor
    tiles.extend([tiles[-1]] * (n_padded - n_tiles))

    for group_start in range(0, n_padded, tta_factor):
        group_tiles = tiles[group_start:group_start + tta_factor]
        batch_imgs = [normalize_for_model(padded[y:y + patch_size, x:x + patch_size]) for y, x in group_tiles]
        batch_tensor = torch.from_numpy(np.stack(batch_imgs, axis=0)).cuda()
        outputs = predict_tta_batch(net, batch_tensor)
        for j in range(tta_factor):
            y, x = tiles[group_start + j]
            score_map[y:y + patch_size, x:x + patch_size] += outputs[j]
            weight_map[y:y + patch_size, x:x + patch_size] += 1
        done = group_start + tta_factor
        print(f'      tiles {done}/{n_padded} done')

    weight_map[weight_map == 0] = 1
    score_map = score_map / weight_map
    return score_map[:h_actual, :w_actual]


def main():
    output_mask_tiff = os.path.join(OUTPUT_DIR, 'caofg_area2_mask.tif')
    output_csv = os.path.join(OUTPUT_DIR, 'caofg_area2_stats.csv')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f'[{time():.1f}] Loading model...')
    net = DinkNet34().cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    net.load_state_dict(torch.load(WEIGHT_PATH))
    net.eval()
    print(f'[{time():.1f}] Model loaded. CUDA: {torch.cuda.get_device_name(0)}')

    print(f'[{time():.1f}] Reading TIFF: {INPUT_TIFF}')
    pil_img = Image.open(INPUT_TIFF)
    w_orig = pil_img.width
    h_orig = pil_img.height
    print(f'[{time():.1f}] Image size: {w_orig} x {h_orig}')

    img_rgba = np.array(pil_img)
    if img_rgba.shape[2] == 4:
        alpha = img_rgba[:, :, 3]
        img_rgb = img_rgba[:, :, :3]
        transparent_mask = (alpha == 0)
        print(f'[{time():.1f}] Transparent pixels (alpha=0): {transparent_mask.sum()} ({100*transparent_mask.sum()/alpha.size:.1f}%)')
    else:
        img_rgb = img_rgba
        transparent_mask = np.zeros((h_orig, w_orig), dtype=bool)
        print(f'[{time():.1f}] No alpha channel found.')

    img_bgr = img_rgb[:, :, ::-1]

    print(f'[{time():.1f}] Sliding-window inference (patch={PATCH_SIZE}, stride={STRIDE}, batch={BATCHSIZE_PER_CARD})...')
    t0 = time()
    score_map = sliding_window_inference(net, img_bgr, PATCH_SIZE, STRIDE)
    print(f'[{time():.1f}] Inference done in {time()-t0:.1f}s')

    print(f'[{time():.1f}] Applying threshold {MASK_THRESHOLD}...')
    binary_mask = np.zeros_like(score_map, dtype=np.uint8)
    binary_mask[score_map > MASK_THRESHOLD] = 255

    valid_mask = ~transparent_mask
    road_pixels = (binary_mask == 255) & valid_mask
    road_pixel_count = int(road_pixels.sum())

    with rasterio.open(INPUT_TIFF) as src:
        crs = src.crs
        transform = src.transform
        res_x = src.res[0]
        res_y = src.res[1]

    pixel_size_m = (abs(res_x) + abs(res_y)) * 0.5
    total_length_m = road_pixel_count * pixel_size_m

    print(f'[{time():.1f}] GSD: {res_x:.6f} m/px (x), {res_y:.6f} m/px (y)')
    print(f'[{time():.1f}] Avg pixel size: {pixel_size_m:.6f} m/px')
    print(f'[{time():.1f}] Road pixels (non-transparent): {road_pixel_count}')
    print(f'[{time():.1f}] Estimated total road length: {total_length_m:.2f} m')

    print(f'[{time():.1f}] Writing GeoTIFF mask ({w_orig}x{h_orig})...')
    mask_3ch = np.stack([binary_mask, binary_mask, binary_mask], axis=-1)
    with rasterio.open(
        output_mask_tiff, 'w',
        driver='GTiff',
        height=h_orig,
        width=w_orig,
        count=3,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
        compress='lzw',
        nodata=0,
    ) as dst:
        dst.write(mask_3ch.transpose(2, 0, 1))
    print(f'[{time():.1f}] GeoTIFF saved: {output_mask_tiff}')

    print(f'[{time():.1f}] Writing CSV: {output_csv}')
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value', 'unit'])
        writer.writerow(['GSD_x', f'{res_x:.8f}', 'm/pixel'])
        writer.writerow(['GSD_y', f'{res_y:.8f}', 'm/pixel'])
        writer.writerow(['pixel_size_avg', f'{pixel_size_m:.8f}', 'm/pixel'])
        writer.writerow(['road_pixel_count', road_pixel_count, 'pixels'])
        writer.writerow(['total_road_length_m', f'{total_length_m:.4f}', 'meters'])
        writer.writerow(['threshold', MASK_THRESHOLD, ''])
        writer.writerow(['patch_size', PATCH_SIZE, 'pixels'])
        writer.writerow(['stride', STRIDE, 'pixels'])
        writer.writerow(['image_width', w_orig, 'pixels'])
        writer.writerow(['image_height', h_orig, 'pixels'])
        writer.writerow(['transparent_pixels', int(transparent_mask.sum()), 'pixels'])
        writer.writerow(['crs', str(crs), ''])

    print(f'\n=== DONE ===')
    print(f'  Mask GeoTIFF: {output_mask_tiff}')
    print(f'  Stats CSV:    {output_csv}')
    print(f'  Road pixels:  {road_pixel_count}')
    print(f'  Road length:  {total_length_m:.2f} m')


if __name__ == '__main__':
    main()
