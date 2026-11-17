#!/usr/bin/env python3
import os
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAKEWORD_MODEL = os.path.join(BASE_DIR, "Sitz_20261014_145414.onnx")

VOICE_COMMAND = "SITZ"
VOICE_THRESHOLD = 0.2
VOICE_VAD = 0.3
VOICE_REQUIRED_CONSECUTIVE = 2
VOICE_COOLDOWN_S = 1.5
AUDIO_RATE = 16000
AUDIO_CHUNK = 1280


class VoiceNode(Node):

  def __init__(self):
    super().__init__("voice_recognition_node")
    self.publisher_ = self.create_publisher(String, "/command/voice", 10)

    if not os.path.exists(WAKEWORD_MODEL):
      self.get_logger().warn(f"Wakeword-Modell fehlt: {WAKEWORD_MODEL}")
      return

    self.stop_event = threading.Event()
    self.listener_thread = threading.Thread(
        target=self.audio_listener_loop, daemon=True
    )
    self.listener_thread.start()
    self.get_logger().info("Voice Recognition Node gestartet.")

  def audio_listener_loop(self):
    try:
      import pyaudio
      from openwakeword.model import Model

      model = Model(
          wakeword_models=[WAKEWORD_MODEL],
          inference_framework="onnx",
          vad_threshold=VOICE_VAD,
      )
      pa = pyaudio.PyAudio()
      stream = pa.open(
          format=pyaudio.paInt16,
          channels=1,
          rate=AUDIO_RATE,
          input=True,
          frames_per_buffer=AUDIO_CHUNK,
      )
    except Exception as e:
      self.get_logger().error(f"Sprachbefehl konnte nicht initialisiert werden: {e}")
      return

    key = os.path.splitext(os.path.basename(WAKEWORD_MODEL))[0]
    hits, last_trigger = 0, 0.0

    try:
      while not self.stop_event.is_set():
        data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
        pred = model.predict(np.frombuffer(data, dtype=np.int16))
        score = pred.get(key, max(pred.values(), default=0.0))

        hits = hits + 1 if score > VOICE_THRESHOLD else 0
        now = time.time()
        if (
            hits >= VOICE_REQUIRED_CONSECUTIVE
            and now - last_trigger > VOICE_COOLDOWN_S
        ):
          last_trigger, hits = now, 0
          msg = String()
          msg.data = VOICE_COMMAND
          self.publisher_.publish(msg)
          self.get_logger().info(f"Sprachbefehl erkannt: {VOICE_COMMAND}")
    finally:
      stream.stop_stream()
      stream.close()
      pa.terminate()

  def destroy_node(self):
    self.stop_event.set()
    super().destroy_node()


def main(args=None):
  rclpy.init(args=args)
  node = VoiceNode()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
  main()