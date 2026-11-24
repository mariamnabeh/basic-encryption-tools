from pathlib import Path
import cv2
import numpy as np


def generate_lorenz_keystream(
    length,
    initial_state=(1.0, 1.0, 1.0),
    dt=0.01,
    discard=500,
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
):
  x = float(initial_state[0])
  y = float(initial_state[1])
  z = float(initial_state[2])
  h2 = 0.5 * dt
  h6 = dt / 6.0

  for _ in range(discard):
    k1x = sigma * (y - x)
    k1y = x * (rho - z) - y
    k1z = x * y - beta * z

    x2 = x + h2 * k1x
    y2 = y + h2 * k1y
    z2 = z + h2 * k1z
    k2x = sigma * (y2 - x2)
    k2y = x2 * (rho - z2) - y2
    k2z = x2 * y2 - beta * z2

    x3 = x + h2 * k2x
    y3 = y + h2 * k2y
    z3 = z + h2 * k2z
    k3x = sigma * (y3 - x3)
    k3y = x3 * (rho - z3) - y3
    k3z = x3 * y3 - beta * z3

    x4 = x + dt * k3x
    y4 = y + dt * k3y
    z4 = z + dt * k3z
    k4x = sigma * (y4 - x4)
    k4y = x4 * (rho - z4) - y4
    k4z = x4 * y4 - beta * z4

    x += h6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y += h6 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    z += h6 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)

  ks = np.empty(length, dtype=np.uint8)
  for i in range(length):
    k1x = sigma * (y - x)
    k1y = x * (rho - z) - y
    k1z = x * y - beta * z

    x2 = x + h2 * k1x
    y2 = y + h2 * k1y
    z2 = z + h2 * k1z
    k2x = sigma * (y2 - x2)
    k2y = x2 * (rho - z2) - y2
    k2z = x2 * y2 - beta * z2

    x3 = x + h2 * k2x
    y3 = y + h2 * k2y
    z3 = z + h2 * k2z
    k3x = sigma * (y3 - x3)
    k3y = x3 * (rho - z3) - y3
    k3z = x3 * y3 - beta * z3

    x4 = x + dt * k3x
    y4 = y + dt * k3y
    z4 = z + dt * k3z
    k4x = sigma * (y4 - x4)
    k4y = x4 * (rho - z4) - y4
    k4z = x4 * y4 - beta * z4

    x += h6 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    y += h6 * (k1y + 2.0 * k2y + 2.0 * k3y + k4y)
    z += h6 * (k1z + 2.0 * k2z + 2.0 * k3z + k4z)

    x_part = int(abs(x) * 1e6)
    y_part = int(abs(y) * 1e6)
    z_part = int(abs(z) * 1e6)
    ks[i] = (x_part ^ (y_part << 5) ^ (z_part << 11)) & 0xFF
  return ks


def xor_encrypt(img_gray, ks=None, initial_state=(1.0, 1.0, 1.0), dt=0.01):
  h, w = img_gray.shape
  flat_p = img_gray.flatten().astype(np.uint8)
  if ks is None:
    ks = generate_lorenz_keystream(h * w, initial_state=initial_state, dt=dt)
  cipher = np.bitwise_xor(flat_p, ks)
  return cipher.reshape((h, w)).astype(np.uint8)


def xor_decrypt(cipher_img, ks=None, initial_state=(1.0, 1.0, 1.0), dt=0.01):
  return xor_encrypt(cipher_img, ks=ks, initial_state=initial_state, dt=dt)


def compute_entropy(img_gray):
  hist = np.bincount(img_gray.flatten(), minlength=256).astype(np.float64)
  probs = hist[hist > 0] / hist.sum()
  return float(-np.sum(probs * np.log2(probs)))


def compute_correlation(img_gray, num_samples=5000):
  arr = img_gray.astype(np.float64)
  h, w = arr.shape
  rng = np.random.default_rng(42)

  r_h = rng.integers(0, h, size=num_samples)
  c_h = rng.integers(0, w - 1, size=num_samples)
  x_h, y_h = arr[r_h, c_h], arr[r_h, c_h + 1]
  corr_h = float(np.corrcoef(x_h, y_h)[0, 1])

  r_v = rng.integers(0, h - 1, size=num_samples)
  c_v = rng.integers(0, w, size=num_samples)
  x_v, y_v = arr[r_v, c_v], arr[r_v + 1, c_v]
  corr_v = float(np.corrcoef(x_v, y_v)[0, 1])

  r_d = rng.integers(0, h - 1, size=num_samples)
  c_d = rng.integers(0, w - 1, size=num_samples)
  x_d, y_d = arr[r_d, c_d], arr[r_d + 1, c_d + 1]
  corr_d = float(np.corrcoef(x_d, y_d)[0, 1])

  return {"Horizontal": corr_h, "Vertical": corr_v, "Diagonal": corr_d}


def compute_metrics(orig_img, enc_e1, enc_e2):
  p = orig_img.astype(np.float64)
  e1 = enc_e1.astype(np.float64)
  e2 = enc_e2.astype(np.float64)

  mae = np.mean(np.abs(p - e1))
  mse = np.mean((p - e1) ** 2)
  diff_mask = (enc_e1 != enc_e2).astype(np.float64)
  npcr = np.mean(diff_mask) * 100.0
  uaci = (np.mean(np.abs(e1 - e2)) / 255.0) * 100.0

  return {"MAE": mae, "MSE": mse, "NPCR (%)": npcr, "UACI (%)": uaci}


def solve():
  base_dir = Path(__file__).resolve().parent
  img_path = base_dir / "image_512.png"
  img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
  assert img is not None, f"File not found: {img_path}"
  img = cv2.resize(img, (512, 512))

  img_mod = img.copy()
  img_mod[256, 256] = np.uint8((int(img_mod[256, 256]) + 1) % 256)

  key_init = (1.0, 1.0, 1.0)
  ks = generate_lorenz_keystream(img.size, initial_state=key_init)
  e1 = xor_encrypt(img, ks=ks)
  e2 = xor_encrypt(img_mod, ks=ks)
  dec = xor_decrypt(e1, ks=ks)
  assert np.array_equal(img, dec)

  cv2.imwrite(str(base_dir / "cipher_xor_e1.png"), e1)
  cv2.imwrite(str(base_dir / "cipher_xor_e2.png"), e2)
  cv2.imwrite(str(base_dir / "decrypted_xor.png"), dec)

  metrics = compute_metrics(img, e1, e2)
  for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

  print(f"Plain Entropy: {compute_entropy(img):.4f}")
  print(f"Cipher Entropy: {compute_entropy(e1):.4f}")

  plain_corr = compute_correlation(img)
  cipher_corr = compute_correlation(e1)
  for axis in ["Horizontal", "Vertical", "Diagonal"]:
    print(f"Plain {axis} Corr: {plain_corr[axis]:.4f}")
    print(f"Cipher {axis} Corr: {cipher_corr[axis]:.4f}")


if __name__ == "__main__":
  solve()