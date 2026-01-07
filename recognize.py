import torch
import torch.nn as nn
import numpy as np
from PIL import Image


class KhmerOCR(nn.Module):
  def __init__(self, num_chars=101, hidden_size=256):
    super(KhmerOCR, self).__init__()
    self.cnn = nn.Sequential(
      self._conv_block(1, 32, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=2, stride=2),
      self._conv_block(32, 64, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=2, stride=2),
      self._conv_block(64, 128, kernel_size=3, stride=1, padding=1),
      self._conv_block(128, 128, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
      self._conv_block(128, 256, kernel_size=3, stride=1, padding=1),
      self._conv_block(256, 256, kernel_size=3, stride=1, padding=1),
      nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)),
    )

    self.lstm1 = nn.LSTM(256, hidden_size, bidirectional=True, batch_first=True)
    self.intermediate_linear = nn.Linear(hidden_size * 2, hidden_size)
    self.lstm2 = nn.LSTM(hidden_size, hidden_size, bidirectional=True, batch_first=True)
    self.fc = nn.Linear(hidden_size * 2, num_chars + 1)

  def _conv_block(self, in_ch, out_ch, **kwargs):
    return nn.Sequential(
      nn.Conv2d(in_ch, out_ch, bias=False, **kwargs),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    features = self.cnn(x)
    features = features.squeeze(2)
    features = features.permute(0, 2, 1)
    rnn_out, _ = self.lstm1(features)
    rnn_out = self.intermediate_linear(rnn_out)
    rnn_out = torch.relu(rnn_out)
    rnn_out, _ = self.lstm2(rnn_out)
    logits = self.fc(rnn_out)
    return logits.permute(1, 0, 2)


TOKENS = (
  "កខគឃងចឆជឈញដឋឌឍណតថទធនបផពភមយរលវឝឞសហឡអឣឤឥឦឧឩឪឫឬឭឮឯឰឱឲឳាិីឹឺុូួើឿៀេែៃោៅំះៈ៉៊់៌៍៎៏័៑្។៕៖ៗ៘៛៝០១២៣៤៥៦៧៨៩៳"
)


def load_image(file: str):
  image = Image.open(file).convert("L")
  image = image.resize((int((image.width / image.height) * 32), 32))
  image = np.array(image) / 255.0
  image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
  return image


def recognize(image_file: str) -> str:
  model = KhmerOCR().eval()
  model.load_state_dict(
    torch.load(
      "model.pt",
      weights_only=True,
      map_location="cpu",
    )
  )

  image = load_image(image_file)

  with torch.no_grad():
    logits = model(image)
    probs = torch.softmax(logits, dim=-1)
    preds = torch.argmax(probs, dim=-1)
    preds = preds.squeeze().tolist()
    decoded_indices = []
    previous_idx = -1
    for idx in preds:
      if idx != previous_idx:
        if idx != 0:
          decoded_indices.append(idx)
      previous_idx = idx
    return "".join([TOKENS[idx - 3] for idx in decoded_indices])


if __name__ == "__main__":
  from argparse import ArgumentParser

  parser = ArgumentParser(description="KhmerOCR script")
  parser.add_argument("image", help="Path to the image file (e.g.) image.jpg")
  args = parser.parse_args()
  text = recognize(args.image)
  print(text)
