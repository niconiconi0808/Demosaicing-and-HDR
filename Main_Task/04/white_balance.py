# === 04/white_balance.py ===
import os, sys, numpy as np, cv2, rawpy
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '02')))
sys.path.append(os.path.abspath(os.path.join(__file__, '..', '..', '03')))
from demosaic import demosaic_image, bayer_pattern_from_raw
from improve_luminosity import improve_luminosity_linear

RAW = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '02', 'IMG_4782.CR3'))

def gray_world(rgb, gain_min=0.4, gain_max=2.5):
    m = rgb.mean(axis=(0,1))
    mg = m.mean()
    gains = np.clip(mg / (m + 1e-9), gain_min, gain_max)
    return rgb * gains

def save_png8(rgb, path, p_lo=0.01, p_hi=99.99):
    # 可视化的线性归一化（0..1）
    a = np.percentile(rgb.mean(axis=2), p_lo)
    b = np.percentile(rgb.mean(axis=2), p_hi)
    vis = np.clip((rgb - a) / (b - a + 1e-12), 0, 1)

    # 8-bit：clip 到 255 并保存
    out8 = (vis * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(out8, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    # 02：去马赛克
    raw = rawpy.imread(RAW)
    pattern = bayer_pattern_from_raw(raw)
    rgb_linear = demosaic_image(RAW, pattern)

    # 03：亮度增强
    rgb_gamma = improve_luminosity_linear(rgb_linear, gamma=0.3)

    # 04：白平衡
    rgb_wb = gray_world(rgb_gamma)

    # 导出
    save_png8(rgb_wb, "IMG_4782_white_balance.png")
    print("done")

    #己完成

