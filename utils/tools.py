import pyglet
from os.path import join

class SimpleImageViewer(object):

  def __init__(self, display=None, save_dir=None):
    self.window = None
    self.isopen = False
    self.display = display
    self.save_dir = save_dir

  def imshow(self, arr, save_img=None):
    if self.window is None:
      height, width, channels = arr.shape
      self.window = pyglet.window.Window(width=width, height=height, display=self.display, caption="THOR Browser")
      self.width = width
      self.height = height
      self.isopen = True

    assert arr.shape == (self.height, self.width, 3), "You passed in an image with the wrong number shape"
    image = pyglet.image.ImageData(self.width, self.height, 'RGB', arr.tobytes(), pitch=self.width * -3)
    self.window.clear()
    self.window.switch_to()
    self.window.dispatch_events()
    image.blit(0,0)
    self.window.flip()
    if self.save_dir and save_img is not None:
      pyglet.image.get_buffer_manager().get_color_buffer().save(
          join(self.save_dir, 'action_{:04d}.png'.format(save_img)))

  def close(self):
    if self.isopen:
      self.window.close()
      self.isopen = False

  def __del__(self):
    self.close()
