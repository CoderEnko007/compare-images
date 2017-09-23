from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2

def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageB.shape[1])
	return err

def compare_image(imageA, imageB, title):
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)

	fig = plt.figure(title)
	plt.suptitle("MSE:%.2f, SSIM:%.2f" % (m, s))

	ax = fig.add_subplot(1, 2, 1)
	ax.imshow(imageA, cmap = plt.cm.gray)
	plt.axis("off")

	ax = fig.add_subplot(1, 2, 2)
	ax.imshow(imageB, cmap = plt.cm.gray)
	plt.axis("off")

	plt.show()

original = cv2.imread("images\\jp_gates_original.png")
contrast = cv2.imread("images\\jp_gates_contrast.png")
shopped = cv2.imread("images\\jp_gates_photoshopped.png")

original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
shopped = cv2.cvtColor(shopped, cv2.COLOR_BGR2GRAY)

fig = plt.figure("image")
images = ("Original", original), ("Contract", contrast), ("Shopped", shopped)

for (i, (name, image)) in enumerate(images):
	ax = fig.add_subplot(1, 3, i + 1)
	ax.set_title(name)
	ax.imshow(image, cmap = plt.cm.gray)
	plt.axis("off")
plt.show()

compare_image(original, original, "Original vs Original")
compare_image(original, contrast, "Original vs Contrast")
compare_image(original, shopped, "Original vs Shopped")
