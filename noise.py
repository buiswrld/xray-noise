import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def add_poisson(image, intensity=0):
  if not intensity == 0:
    print("ALERT: Poisson noise being applied with intensity ", intensity)
  else:
    return image
  image_data = np.array(image) / 255.0 
  PHOTON = 100
  photon_count = PHOTON / (intensity**2)
  scaled_image = image_data * photon_count
  poisson_noisy_image = np.random.poisson(scaled_image) / photon_count
  noisy = np.clip(poisson_noisy_image, 0, 1)
  noisy_image = Image.fromarray((noisy * 255).astype(np.uint8))
  return noisy_image


def add_gaussian(image, intensity=0):
  if not intensity == 0:
    print("ALERT: Electronic noise being applied with intensity ", intensity)
  else:
    return image
  image_data = np.array(image) / 255
  STD_DEV = 0.1 # set baseline at 1%
  sigma = STD_DEV * intensity
  noise = np.random.normal(0, sigma, image_data.shape)
  gaussian_noisy_image = image_data + noise
  noisy = np.clip(gaussian_noisy_image, 0, 1)
  noisy_image = Image.fromarray((noisy * 255).astype(np.uint8))
  return noisy_image


def _show(poisson_noisy_image):
  plt.title('Poisson Noisy X-ray Slice')
  plt.imshow(poisson_noisy_image, cmap='gray')
  plt.axis('off')
  plt.show()
  
if __name__ == "__main__":
  from PIL import Image
  import sys

  # generating samples, using this format:
  # python noise.py <image_path> <quantum_intensity> <electronic_intensity>

  try:
    image_path = sys.argv[1]
    quantum_intensity = int(sys.argv[2])
    electronic_intensity = float(sys.argv[3])
  except:
    print("Invalid format: python noise.py <image_path> <quantum_intensity> <electronic_intensity>")
    sys.exit(1)
  image = Image.open(image_path).convert("L") 
  quantum_noisy = add_poisson(image, intensity=quantum_intensity)
  combined_noisy = add_gaussian(quantum_noisy, intensity=electronic_intensity)
  
  _show(combined_noisy)