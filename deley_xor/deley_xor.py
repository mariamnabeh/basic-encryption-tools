import cv2
import numpy as np


def lorenz_deriv(state, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
  x, y, z = state
  dx_dt = sigma * (y - x)
  dy_dt = x * (rho - z) - y
  dz_dt = x * y - beta * z
  return np.array([dx_dt, dy_dt, dz_dt], dtype=np.float64)


def rk4_step(state, dt, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
  k1 = lorenz_deriv(state, sigma, rho, beta)
  k2 = lorenz_deriv(state + 0.5 * dt * k1, sigma, rho, beta)
  k3 = lorenz_deriv(state + 0.5 * dt * k2, sigma, rho, beta)
  k4 = lorenz_deriv(state + dt * k3, sigma, rho, beta)
  return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_lorenz_keystream(
    length, initial_state=(1.0, 1.0, 1.0), dt=0.01, discard=500
):
  state = np.array(initial_state, dtype=np.float64)
  for _ in range(discard):
    state = rk4_step(state, dt)

  ks = np.empty(length, dtype=np.uint8)
  for i in range(length):
    state = rk4_step(state, dt)
    x_part = int(abs(state[0]) * 1e6)
    y_part = int(abs(state[1]) * 1e6)
    z_part = int(abs(state[2]) * 1e6)
    ks[i] = (x_part ^ (y_part << 5) ^ (z_part << 11)) & 0xFF
  return ks


def xor_with_delay_encrypt(
    img_gray, initial_state=(1.0, 1.0, 1.0), dt=0.01, init_delay=123
):
  h, w = img_gray.shape
  flat_p = img_gray.flatten().astype(np.int32)
  ks = generate_lorenz_keystream(h * w, initial_state=initial_state, dt=dt).astype(
      np.int32
  )

  cipher = np.empty_like(flat_p)
  prev_c = int(init_delay) & 0xFF
  for i in range(len(flat_p)):
    c_val = (flat_p[i] ^ ks[i] ^ prev_c) & 0xFF
    cipher[i] = c_val
    prev_c = c_val

  return cipher.reshape((h, w)).astype(np.uint8)


def xor_with_delay_decrypt(
    cipher_img, initial_state=(1.0, 1.0, 1.0), dt=0.01, init_delay=123
):
  h, w = cipher_img.shape
  flat_c = cipher_img.flatten().astype(np.int32)
  ks = generate_lorenz_keystream(h * w, initial_state=initial_state, dt=dt).astype(
      np.int32
  )

  plain = np.empty_like(flat_c)
  prev_c = int(init_delay) & 0xFF
  for i in range(len(flat_c)):
    c_val = flat_c[i]
    plain[i] = (c_val ^ ks[i] ^ prev_c) & 0xFF
    prev_c = c_val

  return plain.reshape((h, w)).astype(np.uint8)


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
  img = cv2.imread("image_512.png", cv2.IMREAD_GRAYSCALE)
  if img is None:
    img = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    img = cv2.resize(img, (512, 512))
  img_mod = img.copy()
  img_mod[256, 256] = np.uint8((int(img_mod[256, 256]) + 1) % 256)

  key_init = (1.0, 1.0, 1.0)
  e1 = xor_with_delay_encrypt(img, initial_state=key_init)
  e2 = xor_with_delay_encrypt(img_mod, initial_state=key_init)
  dec = xor_with_delay_decrypt(e1, initial_state=key_init)
  assert np.array_equal(img, dec)

  cv2.imwrite("cipher_e1.png", e1)
  cv2.imwrite("cipher_e2.png", e2)
  cv2.imwrite("decrypted.png", dec)

  metrics = compute_metrics(img, e1, e2)
  for k, v in metrics.items():
    print(f"{k}: {v:.4f}")


if __name__ == "__main__":
  solve()