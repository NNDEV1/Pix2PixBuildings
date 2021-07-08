# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.preprocessing.image import array_to_img
from numpy import load
from numpy import expand_dims
from matplotlib import pyplot
 
# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels
 
# load source image
src_image = load_image('/content/base/cmp_b0008.jpg')
print('Loaded', src_image.shape)

with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, "/content/models/model_3000.ckpt")
  print("Model restored.")
  gen_image = model.sample_generator(sess, src_image).squeeze(0)
  gen_image = (gen_image + 1) / 2.0

  plt.imshow((array_to_img(gen_image)))
